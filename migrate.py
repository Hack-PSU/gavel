"""
Gavel schema migrations.

Run manually -- nothing here executes on startup:

    python migrate.py status        # what is applied, what is pending
    python migrate.py upgrade       # apply everything pending
    python migrate.py upgrade --dry-run
    python migrate.py upgrade --database-url postgresql://.../gavel
    python migrate.py upgrade --database-url mysql+pymysql://.../gavel

`--database-url` exists so the same runner can bring up a brand new Cloud SQL
instance, or be pointed at a copy of the on-disk database before cutting over.

This lives at the repository root, and imports nothing from the `gavel`
package, so it runs with only SQLAlchemy and a driver installed. An operator
migrating a database should not need Flask, Celery and the asset pipeline to
import cleanly first.


Why this file exists
--------------------
``db.create_all()`` -- the only schema operation Gavel has ever had -- CREATEs
missing tables and nothing else. It never adds a column to a table that already
exists and never changes a column's type. So every database initialized before
a model changed keeps a schema that silently disagrees with the code.

That is what produced the "notes longer than N characters can't be saved" bug:
``Decision.notes`` is TEXT in the model, but the ``decision`` table predates it
by years, so on a long-lived database the column was either missing or was
hand-added as a ``VARCHAR(n)``. PostgreSQL rejected longer notes with
``StringDataRightTruncation``, which is not a serialization failure, so
``with_retries`` re-raised it and the judge got a bare 500 -- losing the whole
vote, not just the note.


Contract for a migration
------------------------
Every step is *independently idempotent*: it inspects the live schema and does
nothing if the database already matches. The ledger in ``gavel_migration`` is
an optimisation and an audit trail, not the safety mechanism -- which matters
because the existing production database has no ledger, and because a database
restored from a dump may have one that disagrees with reality.

Steps run one at a time, each in its own transaction, under a PostgreSQL
advisory lock so two operators (or an operator and a deploy) cannot race.
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text


# Advisory lock key, so concurrent runs queue instead of colliding.
MIGRATION_LOCK_KEY = 0x6761766D  # b'gavm'

LEDGER_TABLE = 'gavel_migration'


# ------------------------------------------------------------------- dialects
#
# Gavel is moving from PostgreSQL to MySQL (Cloud SQL), so this runner has to
# bring either engine up to the same logical schema. The two differ in ways
# that matter to every step below: catalog tables, how a column's type is
# changed, whether partial indexes exist, and how unbounded text is spelled.

def _dialect(conn):
    return conn.engine.dialect.name


def _is_mysql(conn):
    return _dialect(conn) == 'mysql'


def quote(identifier):
    """
    Quote an identifier for the current statement.

    Gavel has tables named `ignore` and `view` and a column named `key`, all of
    which are reserved words in MySQL. The ORM quotes automatically; the raw
    DDL here has to do it explicitly. Backticks are accepted by MySQL, and
    PostgreSQL accepts double quotes, so each statement is built for the engine
    it will run on.
    """
    return identifier


def _q(conn, identifier):
    return '`%s`' % identifier if _is_mysql(conn) else '"%s"' % identifier


def unbounded_text(conn):
    """The widest text type available. MySQL TEXT stops at 64KB."""
    return 'LONGTEXT' if _is_mysql(conn) else 'TEXT'


# ---------------------------------------------------------------- introspection

def _schema_predicate(conn):
    """Restrict information_schema lookups to the database being migrated.

    MySQL's information_schema spans every database on the server, so without
    this a table of the same name elsewhere would answer the question.
    """
    return 'AND table_schema = DATABASE()' if _is_mysql(conn) else ''


def _column(conn, table, column):
    """Return (data_type, character_maximum_length) or None if absent."""
    return conn.execute(
        text(
            'SELECT data_type, character_maximum_length '
            'FROM information_schema.columns '
            'WHERE table_name = :t AND column_name = :c %s'
            % _schema_predicate(conn)
        ),
        {'t': table, 'c': column},
    ).fetchone()


def _table_exists(conn, table):
    return conn.execute(
        text('SELECT 1 FROM information_schema.tables WHERE table_name = :t %s'
             % _schema_predicate(conn)),
        {'t': table},
    ).fetchone() is not None


def _constraint_exists(conn, name):
    return conn.execute(
        text('SELECT 1 FROM information_schema.table_constraints '
             'WHERE constraint_name = :n %s' % _schema_predicate(conn)),
        {'n': name},
    ).fetchone() is not None


def _index_exists(conn, name, table=None):
    if _is_mysql(conn):
        # MySQL index names are scoped to their table, so the table is part of
        # the question -- unlike PostgreSQL, where relation names are unique
        # within a schema.
        return conn.execute(
            text('SELECT 1 FROM information_schema.statistics '
                 'WHERE index_name = :n AND table_schema = DATABASE() '
                 + ('AND table_name = :t' if table else '')),
            {'n': name, 't': table} if table else {'n': name},
        ).fetchone() is not None
    return conn.execute(
        text('SELECT 1 FROM pg_class WHERE relname = :n AND relkind = :k'),
        {'n': name, 'k': 'i'},
    ).fetchone() is not None


# ------------------------------------------------------------------- primitives

def _ensure_text_column(conn, table, column, log):
    """Ensure `table.column` exists and holds text with no length limit."""
    if not _table_exists(conn, table):
        return
    wide = unbounded_text(conn)
    info = _column(conn, table, column)
    if info is None:
        conn.execute(text('ALTER TABLE %s ADD COLUMN %s %s'
                          % (_q(conn, table), _q(conn, column), wide)))
        log('added %s.%s as %s' % (table, column, wide))
        return

    data_type, max_length = info
    needs_widening = max_length is not None
    if _is_mysql(conn):
        # MySQL reports character_maximum_length for TEXT as well as VARCHAR,
        # so compare the type name instead: anything short of LONGTEXT caps
        # below the "no limit" the models promise.
        needs_widening = (data_type or '').lower() != 'longtext'

    if needs_widening:
        # A bounded column is the trap that broke notes. Widening is a
        # catalog-only change in PostgreSQL; MySQL rewrites the table, which is
        # acceptable at Gavel's data volume.
        if _is_mysql(conn):
            nullable = conn.execute(
                text('SELECT is_nullable FROM information_schema.columns '
                     'WHERE table_name = :t AND column_name = :c '
                     'AND table_schema = DATABASE()'),
                {'t': table, 'c': column},
            ).scalar()
            null_clause = '' if nullable == 'YES' else ' NOT NULL'
            conn.execute(text('ALTER TABLE %s MODIFY COLUMN %s %s%s'
                              % (_q(conn, table), _q(conn, column), wide,
                                 null_clause)))
        else:
            conn.execute(text('ALTER TABLE %s ALTER COLUMN %s TYPE %s'
                              % (_q(conn, table), _q(conn, column), wide)))
        log('widened %s.%s (was %s%s) to %s'
            % (table, column, data_type,
               '(%s)' % max_length if max_length is not None else '', wide))


def _ensure_column(conn, table, column, ddl_type, log, references=None):
    """Ensure `table.column` exists with the given type."""
    if not _table_exists(conn, table):
        return
    if _column(conn, table, column) is not None:
        return

    conn.execute(text('ALTER TABLE %s ADD COLUMN %s %s'
                      % (_q(conn, table), _q(conn, column), ddl_type)))
    log('added %s.%s (%s)' % (table, column, ddl_type))

    if references:
        # MySQL parses an inline REFERENCES clause on ADD COLUMN and then
        # silently ignores it, so the constraint is always added separately.
        ref_table, ref_column = references
        constraint = 'fk_%s_%s' % (table, column)
        try:
            conn.execute(text(
                'ALTER TABLE %s ADD CONSTRAINT %s FOREIGN KEY (%s) '
                'REFERENCES %s (%s)'
                % (_q(conn, table), _q(conn, constraint), _q(conn, column),
                   _q(conn, ref_table), _q(conn, ref_column))))
            log('added foreign key %s.%s -> %s.%s'
                % (table, column, ref_table, ref_column))
        except Exception as e:
            # A pre-tenancy database has rows that do not satisfy the
            # constraint yet; it is added for real in tenancy_constraints once
            # every row has an owner.
            log('deferred foreign key on %s.%s (%s)'
                % (table, column, type(e).__name__))


# -------------------------------------------------------------------- the steps

def notes_are_unbounded(conn, log, opts):
    """
    Make every judge-written free-text column unbounded, and record which
    project a note is actually about.

    A note is about the project the judge was looking at, which may be either
    the winner or the loser of the pair it is stored against. Without
    `notes_item_id` a note can only be guessed at from the pair, which files
    roughly half of them under the wrong project.
    """
    _ensure_text_column(conn, 'decision', 'notes', log)
    _ensure_column(conn, 'decision', 'notes_item_id', 'INTEGER', log,
                   references=('item', 'id'))

    _ensure_text_column(conn, 'skip', 'reason', log)
    # Free text accompanying a skip: required when the reason is "other", and
    # where a judge's notes land when they skip rather than vote. Those notes
    # used to be discarded silently.
    _ensure_text_column(conn, 'skip', 'note', log)


def hackathon_tenancy(conn, log, opts):
    """
    Introduce the hackathon as the tenant that owns all judging data.

    Gavel had no concept of an event, so one database accumulated every
    hackathon forever: last year's projects stayed active and judgeable, judges
    kept their crowd-BT priors and their seen/skipped history across events, and
    "judging is closed" was a single global flag. Scoping everything to a
    hackathon makes each event start clean without anyone remembering to wipe
    the database.
    """
    hackathon_id_type = 'VARCHAR(64)'

    if not _table_exists(conn, 'hackathon'):
        conn.execute(text(
            'CREATE TABLE hackathon ('
            '  id %s NOT NULL PRIMARY KEY,'      # the apiv3 hackathon id
            '  name %s NOT NULL,'
            '  active BOOLEAN NOT NULL DEFAULT FALSE,'
            '  synced TIMESTAMP NULL'
            ')' % (hackathon_id_type, unbounded_text(conn))
        ))
        log('created table hackathon')

    # At most one active hackathon, enforced by the database rather than by
    # whichever worker wrote last.
    if not _index_exists(conn, 'ix_hackathon_single_active', 'hackathon'):
        if _is_mysql(conn):
            # MySQL has no partial indexes. A generated column that is NULL
            # for inactive rows gets the same effect, because a unique index
            # treats NULLs as distinct.
            if _column(conn, 'hackathon', 'active_singleton') is None:
                conn.execute(text(
                    'ALTER TABLE hackathon ADD COLUMN active_singleton '
                    'TINYINT GENERATED ALWAYS AS (CASE WHEN active THEN 1 END) '
                    'STORED'
                ))
            conn.execute(text(
                'CREATE UNIQUE INDEX ix_hackathon_single_active '
                'ON hackathon (active_singleton)'))
        else:
            conn.execute(text(
                'CREATE UNIQUE INDEX ix_hackathon_single_active '
                'ON hackathon ((active)) WHERE active'))
        log('added single-active-hackathon index')

    for table in ('item', 'annotator', 'decision', 'skip'):
        _ensure_column(conn, table, 'hackathon_id', hackathon_id_type, log,
                       references=('hackathon', 'id'))

    # Settings are mostly per-hackathon ("judging is closed"), but a few are
    # global (telemetry timestamps). '' is the global tenant: a composite
    # primary key cannot contain NULL.
    if _table_exists(conn, 'setting'):
        if _column(conn, 'setting', 'hackathon_id') is None:
            conn.execute(text(
                "ALTER TABLE setting ADD COLUMN hackathon_id %s "
                "NOT NULL DEFAULT ''" % hackathon_id_type
            ))
            log("added setting.hackathon_id (NOT NULL DEFAULT '')")

        # MySQL cannot build a primary key over TEXT, so `key` has to be a
        # bounded VARCHAR before it can take part in one.
        key_info = _column(conn, 'setting', 'key')
        if _is_mysql(conn) and key_info and (key_info[0] or '').lower() == 'text':
            conn.execute(text(
                'ALTER TABLE setting MODIFY COLUMN `key` VARCHAR(128) NOT NULL'))
            log('narrowed setting.key to VARCHAR(128) so it can be a key')

        # Replace the key-only primary key with (key, hackathon_id).
        if not _constraint_exists(conn, 'setting_pkey_key_hackathon'):
            if _is_mysql(conn):
                # MySQL primary keys are always named PRIMARY, so there is no
                # separate name to check; dropping is conditional on one
                # existing at all.
                has_pk = conn.execute(text(
                    'SELECT 1 FROM information_schema.table_constraints '
                    "WHERE table_name = 'setting' AND table_schema = DATABASE() "
                    "AND constraint_type = 'PRIMARY KEY'")).fetchone()
                if has_pk:
                    conn.execute(text('ALTER TABLE setting DROP PRIMARY KEY'))
                conn.execute(text(
                    'ALTER TABLE setting ADD PRIMARY KEY (`key`, hackathon_id)'))
            else:
                conn.execute(text(
                    'ALTER TABLE setting DROP CONSTRAINT IF EXISTS setting_pkey'))
                conn.execute(text(
                    'ALTER TABLE setting ADD CONSTRAINT setting_pkey_key_hackathon '
                    'PRIMARY KEY (key, hackathon_id)'))
            log('repointed setting primary key to (key, hackathon_id)')


def adopt_existing_data(conn, log, opts):
    """
    Attribute the rows that already exist to the event they came from.

    Every item, annotator, decision and skip written before tenancy existed
    belongs to a real hackathon -- whichever one was running at the time.
    Gavel has no way to work out which: the rows carry timestamps but nothing
    identifying the event, and the HackPSU API's hackathon list needs a
    credential Gavel does not hold. So the operator names it.

    An earlier version of this step invented a hackathon with the id 'legacy'.
    That is wrong: the id is meant to match the HackPSU API's hackathon id, and
    a made-up one silently disconnects a whole event's results from the system
    of record. Better to stop and ask.

    The adopted hackathon is created inactive -- adopting history must not make
    a previous event's projects judgeable again.

    Safe to re-run: only rows with a NULL hackathon_id are touched, and a
    database with none (a fresh Cloud SQL instance) needs no id at all.
    """
    orphans = 0
    for table in ('item', 'annotator', 'decision', 'skip'):
        if not _table_exists(conn, table):
            continue
        orphans += conn.execute(
            text('SELECT count(*) FROM %s WHERE hackathon_id IS NULL' % table)
        ).scalar()

    if not orphans:
        log('no rows need adopting')
        return

    hackathon_id = opts.get('adopt_hackathon')
    if not hackathon_id:
        # `time` is a reserved word in MySQL, and ::date is PostgreSQL-only.
        span = conn.execute(text(
            'SELECT DATE(MIN(%s)), DATE(MAX(%s)) FROM decision'
            % (_q(conn, 'time'), _q(conn, 'time'))
        )).fetchone() if _table_exists(conn, 'decision') else (None, None)
        raise RuntimeError(
            '%d row(s) predate hackathon tenancy and must be attributed to the '
            'event they came from.\n\n'
            '  Those decisions run from %s to %s.\n\n'
            'Find the matching hackathon id (GET /hackathons on the HackPSU '
            'API, or the admin UI) and re-run with:\n'
            '    python migrate.py upgrade --adopt-hackathon <id> '
            '--adopt-name "<name>"\n\n'
            'The hackathon is created inactive, so adopting this data will not '
            'make its projects judgeable again.'
            % (orphans, span[0], span[1])
        )

    name = opts.get('adopt_name') or hackathon_id

    exists = conn.execute(
        text('SELECT 1 FROM hackathon WHERE id = :id'), {'id': hackathon_id}
    ).fetchone()
    if not exists:
        conn.execute(
            text('INSERT INTO hackathon (id, name, active) '
                 'VALUES (:id, :name, FALSE)'),
            {'id': hackathon_id, 'name': name},
        )
        log('created hackathon %r (%s), inactive' % (hackathon_id, name))

    for table in ('item', 'annotator', 'decision', 'skip'):
        if not _table_exists(conn, table):
            continue
        count = conn.execute(
            text('UPDATE %s SET hackathon_id = :id WHERE hackathon_id IS NULL'
                 % table),
            {'id': hackathon_id},
        ).rowcount
        if count:
            log('adopted %d row(s) in %s' % (count, table))

    # Existing settings were global; "closed" belongs to the adopted event.
    if _table_exists(conn, 'setting'):
        count = conn.execute(
            text("UPDATE setting SET hackathon_id = :id "
                 "WHERE hackathon_id = '' AND key <> :telemetry"),
            {'id': hackathon_id, 'telemetry': 'telemetry_sent_time'},
        ).rowcount
        if count:
            log('scoped %d setting(s) to %s' % (count, hackathon_id))


def tenancy_constraints(conn, log, opts):
    """
    Lock tenancy down once every row has an owner.

    Split from the adoption step so that a database which still has orphans
    fails here loudly rather than half-applying a NOT NULL.
    """
    hackathon_id_type = 'VARCHAR(64)'

    for table in ('item', 'annotator', 'decision', 'skip'):
        if not _table_exists(conn, table):
            continue
        orphans = conn.execute(
            text('SELECT count(*) FROM %s WHERE hackathon_id IS NULL'
                 % _q(conn, table))
        ).scalar()
        if orphans:
            raise RuntimeError(
                '%s still has %d row(s) with no hackathon_id -- run the '
                'adopt_existing_data step first' % (table, orphans)
            )
        info = conn.execute(
            text('SELECT is_nullable FROM information_schema.columns '
                 'WHERE table_name = :t AND column_name = :c %s'
                 % _schema_predicate(conn)),
            {'t': table, 'c': 'hackathon_id'},
        ).fetchone()
        if info and info[0] == 'YES':
            if _is_mysql(conn):
                conn.execute(text('ALTER TABLE %s MODIFY COLUMN hackathon_id '
                                  '%s NOT NULL'
                                  % (_q(conn, table), hackathon_id_type)))
            else:
                conn.execute(text('ALTER TABLE %s ALTER COLUMN hackathon_id '
                                  'SET NOT NULL' % _q(conn, table)))
            log('%s.hackathon_id is now NOT NULL' % table)

        # The foreign key may have been deferred while rows were still
        # unowned; now that they all have an owner it can be created.
        constraint = 'fk_%s_hackathon_id' % table
        if not _constraint_exists(conn, constraint):
            try:
                conn.execute(text(
                    'ALTER TABLE %s ADD CONSTRAINT %s FOREIGN KEY '
                    '(hackathon_id) REFERENCES hackathon (id)'
                    % (_q(conn, table), _q(conn, constraint))))
                log('added foreign key %s.hackathon_id -> hackathon.id' % table)
            except Exception:
                pass  # already present under a different name

    # One judge row per person per event: this is what makes a judge's priors,
    # assignment and seen/skipped history reset when a new hackathon starts.
    if _table_exists(conn, 'annotator') and \
            not _index_exists(conn, 'ix_annotator_email_hackathon', 'annotator'):
        dupes = conn.execute(text(
            'SELECT count(*) FROM (SELECT email, hackathon_id FROM annotator '
            'GROUP BY email, hackathon_id HAVING count(*) > 1) d'
        )).scalar()
        if dupes:
            log('WARNING: %d duplicate (email, hackathon) judge(s) -- skipping '
                'unique index; deduplicate then re-run' % dupes)
        else:
            conn.execute(text(
                'CREATE UNIQUE INDEX ix_annotator_email_hackathon '
                'ON annotator (email, hackathon_id)'))
            log('added unique index on annotator (email, hackathon_id)')

    # Project sync matches on name; without this, two workers racing the same
    # sync both insert the same project.
    if _table_exists(conn, 'item') and \
            not _index_exists(conn, 'ix_item_name_hackathon', 'item'):
        dupes = conn.execute(text(
            'SELECT count(*) FROM (SELECT %s, hackathon_id FROM item '
            'GROUP BY %s, hackathon_id HAVING count(*) > 1) d'
            % (_q(conn, 'name'), _q(conn, 'name'))
        )).scalar()
        if dupes:
            log('WARNING: %d duplicate (name, hackathon) project(s) -- skipping '
                'unique index; deduplicate then re-run' % dupes)
        else:
            if _is_mysql(conn):
                # item.name is unbounded text, which MySQL cannot index whole;
                # a 191-character prefix is well beyond any real project name
                # and stays inside the 3072-byte index limit for utf8mb4.
                conn.execute(text(
                    'CREATE UNIQUE INDEX ix_item_name_hackathon '
                    'ON item (hackathon_id, `name`(191))'))
            else:
                conn.execute(text(
                    'CREATE UNIQUE INDEX ix_item_name_hackathon '
                    'ON item (name, hackathon_id)'))
            log('added unique index on item (name, hackathon_id)')


# Ordered. Append new steps; never renumber or reorder existing ones.
MIGRATIONS = [
    ('0001_notes_are_unbounded', notes_are_unbounded),
    ('0002_hackathon_tenancy', hackathon_tenancy),
    ('0003_adopt_existing_data', adopt_existing_data),
    ('0004_tenancy_constraints', tenancy_constraints),
]


# ------------------------------------------------------------------- the runner

def _take_lock(conn):
    """
    Serialize migration runs, so two operators cannot apply steps at once.

    PostgreSQL's transaction-scoped advisory lock releases itself at commit.
    MySQL has no transaction-scoped equivalent, so GET_LOCK is taken with a
    zero timeout and released explicitly by _release_lock().
    """
    if _is_mysql(conn):
        return bool(conn.execute(
            text('SELECT GET_LOCK(:n, 0)'), {'n': 'gavel_migration'}
        ).scalar())
    return bool(conn.execute(
        text('SELECT pg_try_advisory_xact_lock(:k)'),
        {'k': MIGRATION_LOCK_KEY},
    ).scalar())


def _release_lock(conn):
    if _is_mysql(conn):
        conn.execute(text('SELECT RELEASE_LOCK(:n)'), {'n': 'gavel_migration'})


def _insert_ledger_sql(conn):
    """Record a step as applied, ignoring a row that is already there."""
    if _is_mysql(conn):
        return ('INSERT IGNORE INTO %s (name) VALUES (:n)' % LEDGER_TABLE)
    return ('INSERT INTO %s (name) VALUES (:n) '
            'ON CONFLICT (name) DO NOTHING' % LEDGER_TABLE)


def _ensure_ledger(conn):
    # VARCHAR, not TEXT: MySQL cannot make a primary key from unbounded text.
    # CURRENT_TIMESTAMP is spelled the same on both engines; now() is not.
    conn.execute(text(
        'CREATE TABLE IF NOT EXISTS %s ('
        '  name VARCHAR(128) NOT NULL PRIMARY KEY,'
        '  applied TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP'
        ')' % LEDGER_TABLE
    ))


def _applied(conn):
    return {
        row[0] for row in
        conn.execute(text('SELECT name FROM %s' % LEDGER_TABLE)).fetchall()
    }


def status(engine):
    with engine.begin() as conn:
        missing = [t for t in BASELINE_TABLES if not _table_exists(conn, t)]
        if missing:
            print('  baseline tables missing: %s' % ', '.join(missing))
            print('  run `python initialize.py` before migrating\n')
        _ensure_ledger(conn)
        applied = _applied(conn)
    for name, _fn in MIGRATIONS:
        print('  [%s] %s' % ('x' if name in applied else ' ', name))
    pending = [n for n, _ in MIGRATIONS if n not in applied]
    print('\n%d applied, %d pending' % (len(MIGRATIONS) - len(pending),
                                        len(pending)))
    return pending


# Tables that `python initialize.py` (db.create_all) is responsible for
# creating. Migrations move that baseline forward; they do not establish it.
BASELINE_TABLES = ('item', 'annotator', 'decision', 'setting')


def _check_baseline(conn):
    """
    Refuse to run against a database that has no Gavel tables yet.

    Every step here is conditional on the live schema, so against an empty
    database they would all no-op *and still be recorded as applied* -- leaving
    a database that looks migrated but never received the indexes and
    constraints. Creating the tables is `initialize.py`'s job; this runs after.
    """
    missing = [t for t in BASELINE_TABLES if not _table_exists(conn, t)]
    if len(missing) == len(BASELINE_TABLES):
        raise RuntimeError(
            'No Gavel tables found. Create the baseline schema first:\n'
            '    python initialize.py\n'
            'then re-run this migration.'
        )
    if missing:
        raise RuntimeError(
            'Database is partially initialized (missing: %s). Run '
            '`python initialize.py` to complete the baseline, then re-run.'
            % ', '.join(missing)
        )


def upgrade(engine, dry_run=False, opts=None):
    opts = opts or {}
    with engine.begin() as conn:
        _check_baseline(conn)
        _ensure_ledger(conn)
        applied = _applied(conn)

    pending = [(n, f) for n, f in MIGRATIONS if n not in applied]
    if not pending:
        print('Nothing to do; database is up to date.')
        return 0

    if dry_run:
        print('Would apply:')
        for name, _fn in pending:
            print('  %s' % name)
        return 0

    for name, fn in pending:
        print('==> %s' % name)
        changes = []

        def log(message, _changes=changes):
            _changes.append(message)
            print('    %s' % message)

        # Each step gets its own transaction: a failure leaves earlier steps
        # applied and recorded, so a re-run picks up where it stopped.
        with engine.begin() as conn:
            if not _take_lock(conn):
                print('    another migration run holds the lock; stopping')
                return 1
            try:
                fn(conn, log, opts)
                conn.execute(text(_insert_ledger_sql(conn)), {'n': name})
            finally:
                _release_lock(conn)
        if not changes:
            print('    (already satisfied)')

    print('\nDone.')
    return 0


def _resolve_url(explicit):
    if explicit:
        return explicit
    # Same resolution order the app uses, so `python migrate.py` with no
    # arguments targets the database the app would open.
    for var in ('DATABASE_URL', 'DB_URI'):
        value = os.environ.get(var)
        if value:
            return value.replace('postgres://', 'postgresql://', 1) \
                if value.startswith('postgres://') else value
    return 'postgresql://localhost/gavel'


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python migrate.py',
        description='Apply Gavel schema migrations. Never runs automatically.',
    )
    parser.add_argument('command', choices=('status', 'upgrade'))
    parser.add_argument(
        '--database-url', default=None,
        help='Target database. Defaults to DATABASE_URL, then DB_URI. Point '
             'this at Cloud SQL to prepare a new instance.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='List the migrations that would run, and change nothing.',
    )
    parser.add_argument(
        '--adopt-hackathon', default=os.environ.get('LEGACY_HACKATHON_ID'),
        help='Hackathon id that pre-tenancy rows belong to. Use the id from '
             'the HackPSU API so Gavel and the API agree on the event.',
    )
    parser.add_argument(
        '--adopt-name', default=os.environ.get('LEGACY_HACKATHON_NAME'),
        help='Display name for --adopt-hackathon.',
    )
    args = parser.parse_args(argv)

    url = _resolve_url(args.database_url)
    # Never print credentials back at the operator.
    safe = url.split('@')[-1] if '@' in url else url
    print('Target: ...@%s\n' % safe)

    engine = create_engine(url)
    if engine.dialect.name not in ('postgresql', 'mysql'):
        print('Supported engines are PostgreSQL and MySQL; got %r.'
              % engine.dialect.name)
        return 2
    print('Engine: %s\n' % engine.dialect.name)

    try:
        if args.command == 'status':
            status(engine)
            return 0
        return upgrade(engine, dry_run=args.dry_run, opts={
            'adopt_hackathon': args.adopt_hackathon,
            'adopt_name': args.adopt_name,
        })
    except RuntimeError as e:
        print('ERROR: %s' % e)
        return 2


if __name__ == '__main__':
    sys.exit(main())
