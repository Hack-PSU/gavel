"""
Copy Gavel's data between database engines.

Built for the PostgreSQL -> MySQL move: production currently runs PostgreSQL
inside the application container on a volume that Cloud Run discards, and the
destination is a managed Cloud SQL (MySQL) instance alongside the rest of
HackPSU's data.

    python transfer.py \\
        --source postgresql://gavel@localhost/gavel \\
        --target 'mysql+pymysql://gavel:pw@127.0.0.1:3306/gavel'

    python transfer.py --source ... --target ... --dry-run
    python transfer.py --source ... --target ... --truncate

A dump-and-load will not do this job. `pg_dump` emits PostgreSQL SQL that MySQL
cannot parse, and the two disagree on booleans, identifier quoting (Gavel has
tables named `ignore` and `view` and a column named `key`, all reserved words
in MySQL) and sequence handling. This reads rows through SQLAlchemy and writes
them through the target's own dialect, so each engine renders its own types.

Preconditions: the target already has the schema.

    DATABASE_URL=<target> python initialize.py
    python migrate.py upgrade --database-url <target> \\
        --adopt-hackathon <id> --adopt-name "<name>"

Both sides should be migrated to the same point before transferring, or the
column sets will not line up.
"""

import argparse
import decimal
import hashlib
import sys

from sqlalchemy import MetaData, Table, create_engine, func, select


# Parents before children, so foreign keys are satisfiable as rows land.
# annotator.next_id/prev_id point at item, so item precedes annotator.
TABLE_ORDER = [
    'hackathon',
    'item',
    'annotator',
    'decision',
    'skip',
    'setting',
    'view',
    'ignore',
]

# Rows per INSERT. Large enough to keep round trips down, small enough that a
# single statement stays well inside MySQL's max_allowed_packet even when a
# judge has written a very long note.
BATCH_SIZE = 500


def _reflect(engine, name):
    metadata = MetaData()
    try:
        table = Table(name, metadata, autoload_with=engine)
    except Exception:
        return None

    # SQLAlchemy reflects MySQL DOUBLE with asdecimal=True, and the driver
    # rounds to ten significant digits on the way into a Decimal -- so a value
    # stored identically on both engines comes back differing in the eleventh
    # digit. Ask for plain floats instead; the bits on disk are the same.
    for column in table.columns:
        if getattr(column.type, 'asdecimal', False):
            column.type.asdecimal = False
    return table


def _generated_columns(engine, table_name):
    """
    Columns the database computes for itself, which must not be inserted.

    The single-active-hackathon guard on MySQL is a generated column, and
    writing to one is an error rather than a no-op.
    """
    if engine.dialect.name != 'mysql':
        return set()
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s "
            "AND extra LIKE '%%GENERATED%%'",
            (table_name,),
        ).fetchall()
    return {r[0] for r in rows}


def _row_count(engine, table):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar()


def _fk_checks(engine, enabled):
    """
    Toggle foreign key enforcement on the target for the duration of the copy.

    TABLE_ORDER already satisfies the constraints, but annotator.next_id and
    prev_id point at items that may themselves be mid-copy, and a row that
    references a deleted project (which pre-tenancy data contains) would
    otherwise abort the whole transfer. Checks are restored and the result
    verified afterwards.
    """
    if engine.dialect.name != 'mysql':
        return
    with engine.begin() as conn:
        conn.exec_driver_sql('SET FOREIGN_KEY_CHECKS = %d' % (1 if enabled else 0))


def _has_integer_identity(table, column='id'):
    """
    Whether `column` is a database-generated integer key.

    Only those need their counter advanced after explicit ids are inserted.
    hackathon.id is a VARCHAR carrying the HackPSU id, and the join tables have
    no surrogate key at all.
    """
    if column not in table.columns:
        return False
    col = table.columns[column]
    try:
        return isinstance(col.type.python_type, type) and \
            col.type.python_type is int
    except NotImplementedError:
        return False


def _reset_autoincrement(engine, table_name, column='id'):
    """
    Point MySQL's counter past the ids that were just inserted explicitly.

    Without this the next INSERT reuses id 1 and collides immediately.
    PostgreSQL sequences need the same treatment, via setval.
    """
    with engine.begin() as conn:
        if engine.dialect.name == 'mysql':
            highest = conn.exec_driver_sql(
                'SELECT COALESCE(MAX(`%s`), 0) FROM `%s`' % (column, table_name)
            ).scalar()
            conn.exec_driver_sql(
                'ALTER TABLE `%s` AUTO_INCREMENT = %d'
                % (table_name, int(highest) + 1)
            )
        elif engine.dialect.name == 'postgresql':
            conn.exec_driver_sql(
                "SELECT setval(pg_get_serial_sequence('%s', '%s'), "
                "GREATEST((SELECT COALESCE(MAX(%s), 1) FROM %s), 1))"
                % (table_name, column, column, table_name)
            )


def _normalise(value):
    """
    Make a value comparable across engines.

    MySQL has no boolean type -- BOOLEAN is an alias for TINYINT(1) -- so a
    column PostgreSQL returns as True/False comes back as 1/0. The ORM coerces
    this correctly at runtime; a raw row-by-row comparison has to do it here,
    or every table with a flag looks like it failed to copy.

    Decimal/float and datetime values are likewise rendered to a canonical form
    so that driver-level type differences do not read as data differences.
    """
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    if isinstance(value, decimal.Decimal):
        # SQLAlchemy reflects MySQL DOUBLE with asdecimal=True, so a value
        # PostgreSQL hands back as a float arrives here as a Decimal. The
        # stored bits are identical; only the Python type differs.
        value = float(value)
    if isinstance(value, float):
        # Enough precision to catch a real difference, not so much that the
        # engines' last-bit rounding registers as one.
        return round(value, 9)
    return value


def content_digest(engine, table, columns, order_by):
    """A hash of a table's contents, stable across engines."""
    query = select(*[table.c[c] for c in columns]).order_by(
        *[table.c[c] for c in order_by])
    digest = hashlib.sha256()
    with engine.connect() as conn:
        result = conn.execute(query)
        while True:
            chunk = result.fetchmany(BATCH_SIZE)
            if not chunk:
                break
            for row in chunk:
                digest.update(repr(tuple(_normalise(v) for v in row)).encode())
    return digest.hexdigest()[:16]


def copy_table(source_engine, target_engine, name, truncate, dry_run, log):
    source = _reflect(source_engine, name)
    target = _reflect(target_engine, name)

    if source is None:
        log('%-11s skipped (not in source)' % name)
        return None
    if target is None:
        log('%-11s SKIPPED -- missing in target; run initialize.py and '
            'migrate.py against the target first' % name)
        return ('missing', name)

    source_count = _row_count(source_engine, source)
    existing = _row_count(target_engine, target)

    if existing and not truncate:
        log('%-11s REFUSED -- target already has %d row(s); pass --truncate '
            'to replace' % (name, existing))
        return ('occupied', name)

    # Only columns both sides agree on, minus anything the target computes.
    generated = _generated_columns(target_engine, name)
    columns = [c.name for c in source.columns
               if c.name in target.columns and c.name not in generated]
    dropped = [c.name for c in source.columns if c.name not in target.columns]
    if dropped:
        log('%-11s note: source-only columns not copied: %s'
            % (name, ', '.join(dropped)))

    if dry_run:
        log('%-11s would copy %d row(s) (%d columns)'
            % (name, source_count, len(columns)))
        return None

    if existing:
        with target_engine.begin() as conn:
            conn.execute(target.delete())

    copied = 0
    with source_engine.connect() as src:
        result = src.execute(select(*[source.c[c] for c in columns]))
        while True:
            chunk = result.fetchmany(BATCH_SIZE)
            if not chunk:
                break
            payload = [dict(zip(columns, row)) for row in chunk]
            with target_engine.begin() as conn:
                conn.execute(target.insert(), payload)
            copied += len(payload)

    if _has_integer_identity(target, 'id'):
        _reset_autoincrement(target_engine, name)

    final = _row_count(target_engine, target)
    status = 'OK' if final == source_count else 'MISMATCH'
    log('%-11s %d -> %d rows  [%s]' % (name, source_count, final, status))
    return None if status == 'OK' else ('mismatch', name)


def verify(source_engine, target_engine, log):
    """
    Independently re-check the copy, after constraints are back on.

    Row counts alone would not catch a column that silently lost its contents,
    so this also compares a content hash of every column both sides share.
    """
    log('')
    log('Verification (foreign keys re-enabled):')
    problems = []
    for name in TABLE_ORDER:
        source = _reflect(source_engine, name)
        target = _reflect(target_engine, name)
        if source is None or target is None:
            continue

        a = _row_count(source_engine, source)
        b = _row_count(target_engine, target)
        if a != b:
            problems.append(name)
            log('  %-11s source=%-7d target=%-7d ROW COUNT MISMATCH'
                % (name, a, b))
            continue

        generated = _generated_columns(target_engine, name)
        columns = [c.name for c in source.columns
                   if c.name in target.columns and c.name not in generated]
        order_by = ['id'] if 'id' in columns else sorted(columns)

        src_digest = content_digest(source_engine, source, columns, order_by)
        dst_digest = content_digest(target_engine, target, columns, order_by)
        matched = src_digest == dst_digest
        if not matched:
            problems.append(name)
        log('  %-11s %-7d rows  %s %s %s'
            % (name, a, src_digest, '==' if matched else '!=', dst_digest))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python transfer.py',
        description='Copy Gavel data between database engines.',
    )
    parser.add_argument('--source', required=True, help='Source database URL.')
    parser.add_argument('--target', required=True, help='Target database URL.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be copied, and change nothing.')
    parser.add_argument('--truncate', action='store_true',
                        help='Replace rows already present in the target.')
    args = parser.parse_args(argv)

    def log(message):
        print(message)

    source_engine = create_engine(args.source)
    target_engine = create_engine(args.target)

    safe = lambda url: url.split('@')[-1] if '@' in url else url
    log('Source: %s (%s)' % (safe(args.source), source_engine.dialect.name))
    log('Target: %s (%s)' % (safe(args.target), target_engine.dialect.name))
    log('')

    if not args.dry_run:
        _fk_checks(target_engine, False)
    failures = []
    try:
        for name in TABLE_ORDER:
            problem = copy_table(source_engine, target_engine, name,
                                 args.truncate, args.dry_run, log)
            if problem:
                failures.append(problem)
    finally:
        if not args.dry_run:
            _fk_checks(target_engine, True)

    if args.dry_run:
        log('\nDry run; nothing was written.')
        return 0

    mismatched = verify(source_engine, target_engine, log)
    if failures or mismatched:
        log('\nFAILED: %s' % ', '.join(
            sorted({n for _, n in failures} | set(mismatched))))
        return 1

    log('\nTransfer complete; all row counts match.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
