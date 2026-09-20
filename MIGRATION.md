# Database migrations and the Cloud SQL move

## Local development against a production snapshot

`local/` is gitignored and holds a read-only snapshot of the production
database plus the environment pulled from the production container.

```sh
./local/restore-dev-db.sh      # reset gavel_dev from the newest snapshot
./local/run-dev.sh             # serve on http://localhost:5001
```

Port 5001, because macOS binds 5000 to the AirPlay Receiver.

### Logging in locally

The auth server sets its session cookie on `.hackpsu.org` only. For any other
origin it cannot, so `buildReturnUrl()` appends the token to the redirect as
`?authToken=` instead. Gavel only ever read the cookie, so that half of the
handshake was missing and logging in from localhost just looped back to the
login page. It now accepts the handoff: the token is verified against the auth
server, stored as a cookie, and stripped from the URL by a redirect so it does
not linger in history or a `Referer` header. This applies to Vercel previews
too, not just localhost.

Visit `http://localhost:5001/`, get bounced to `auth.hackpsu.org`, log in, and
you land back with a working session.

### The SCSS build

`local/venv` uses **libsass**, not the pinned pyScss. pyScss 1.3.7 imports
`collections.Iterable`, removed in Python 3.10, and 1.4.0 fails to parse this
stylesheet's `mix()` calls. Production runs Python 3.9, so `SCSS_FILTER`
defaults to `pyscss` and production is unchanged; `local/.env.dev` sets
`SCSS_FILTER=libsass`. Note that having a broken pyScss *installed* also breaks
webassets, which imports every filter module at startup -- so `local/venv` has
it uninstalled rather than merely unused.

The dumps in `local/backups/` are `chmod 444` and checksummed; the restore
script verifies the digest and refuses to run if it does not match. Nothing
writes to them, so local testing can be as destructive as it likes. To take a
fresh snapshot:

```sh
gcloud compute ssh docker-services --project=hackpsu-408118 \
  --zone=us-east1-c --tunnel-through-iap \
  --command="docker exec gavel pg_dump -U gavel --no-owner --no-privileges gavel | gzip -9" \
  > local/backups/gavel-prod-$(date +%Y%m%d-%H%M%S).sql.gz
```

`local/.env.dev` carries the production values for shared config and
credentials, and overrides what must differ locally: `DATABASE_URL` points at
`gavel_dev`, `ENABLE_PROJECT_SYNC` is off (it writes projects and would talk to
the live API), and `HACKATHON_ID=legacy` pins the tenant to the snapshot the
backfill creates.

## Why there is a migration runner at all

`db.create_all()` — the only schema operation Gavel has ever had — creates
missing *tables*. It never adds a column to a table that already exists and
never changes a column's type. Every database initialized before a model
changed therefore runs on a schema that silently disagrees with the code.

This is the shape of bug that produces "notes longer than N characters can't be
saved": a model that says `TEXT` over a column that is still a bounded
`VARCHAR(n)`. PostgreSQL rejects the longer value with
`StringDataRightTruncation`, which is not a serialization failure, so
`with_retries` re-raises it and the judge gets a bare 500 — losing the whole
vote, not just the note.

**On the current production database this is not what is happening.** Inspecting
it directly showed `decision.notes` is already `text`, with the longest stored
note at 364 characters, and no length limit at the proxy either. `skip.reason`
*was* still `VARCHAR(100)`, and the migration widened it. Either the reported
failure predates a redeploy that recreated the container's volume — `start.sh`
runs `initdb` on an empty volume and `create_all()` then builds `notes` as
`TEXT`, erasing the evidence — or the report was really about notes being
dropped on skip, or filed against the wrong project. Both of those were real,
and both are fixed.

`migrate.py` closes that gap. It is **never run automatically**: no startup
hook, no deploy step. An operator runs it.

## Using it

```sh
python migrate.py status                       # what is applied, what is pending
python migrate.py upgrade --dry-run            # show the plan, change nothing
python migrate.py upgrade                      # apply
python migrate.py upgrade --database-url postgresql://user:pw@host/gavel
```

It lives at the repository root and imports nothing from the `gavel` package,
so it needs only SQLAlchemy and psycopg2 — migrating a database does not
require Flask, Celery and the asset pipeline to import cleanly first.

Properties worth relying on:

- **Every step is independently idempotent.** Each inspects the live schema and
  does nothing if the database already matches. The ledger in `gavel_migration`
  is an audit trail and an optimisation, not the safety mechanism — which
  matters because the existing production database has no ledger.
- **Steps are individually transactional.** A failure leaves earlier steps
  applied and recorded; re-running picks up where it stopped.
- **Concurrent runs are safe.** A PostgreSQL advisory lock makes a second run
  wait rather than collide.
- **It refuses to run against an empty database.** Creating the baseline is
  `initialize.py`'s job. Without this guard every step would no-op against a
  table-less database *and still be recorded as applied*, leaving something
  that looks migrated but never received its indexes and constraints.

## Moving from PostgreSQL to MySQL (Cloud SQL)

Gavel runs on either engine. The models use types that mean the same thing on
both (`gavel/models/types.py`), the retry logic recognises both engines'
serialization failures, and `migrate.py` emits DDL for whichever it is pointed
at.

Three engine differences would silently corrupt the data if left to defaults,
and are handled explicitly:

| | Default behaviour | What Gavel does |
|---|---|---|
| Notes | MySQL `TEXT` caps at 64KB and truncates | `LONGTEXT` |
| Scores | SQLAlchemy `Float` is single precision on MySQL, so `mu` 0.0799322734941707 becomes 0.0799323 -- enough to reorder close projects | `DOUBLE` |
| Timestamps | MySQL `DATETIME` keeps whole seconds and *rounds*: 18:39:45.720777 becomes 18:39:46 | `DATETIME(6)` |

MySQL also reserves `ignore`, `view` and `key`, all of which Gavel uses as
identifiers. The ORM quotes them; the raw DDL in `migrate.py` quotes per
dialect.

### The cutover

```sh
export SRC='postgresql://gavel@localhost/gavel'
export DST='mysql+pymysql://gavel:PASSWORD@/gavel?unix_socket=/cloudsql/PROJECT:REGION:INSTANCE'

# 1. Bring both sides to the same schema point, or the columns will not line up.
python migrate.py upgrade --database-url "$SRC" \
    --adopt-hackathon <id> --adopt-name "<name>"
DATABASE_URL="$DST" python initialize.py
python migrate.py upgrade --database-url "$DST" \
    --adopt-hackathon <id> --adopt-name "<name>"

# 2. Rehearse, then copy.
python transfer.py --source "$SRC" --target "$DST" --dry-run
python transfer.py --source "$SRC" --target "$DST"

# 3. Repoint DATABASE_URL at Cloud SQL and redeploy.
```

`transfer.py` reads rows through SQLAlchemy and writes them through the
target's dialect, so each engine renders its own types -- `pg_dump` output is
PostgreSQL SQL that MySQL cannot parse. It copies parents before children,
suspends foreign key checks for the copy, resets MySQL's `AUTO_INCREMENT` past
the explicit ids, and then verifies: row counts *and* a per-table content hash
of every shared column, with booleans and floats normalised so that a
representation difference does not read as a data difference.

It refuses to write into a non-empty target unless `--truncate` is given.

## Moving from the in-container volume to Cloud SQL

`Dockerfile.prod` runs PostgreSQL inside the application container and declares
`VOLUME ["/var/lib/postgresql/data"]`. Cloud Run ignores that: the filesystem is
in-memory and discarded, so judging data does not survive a cold start. Moving
to a managed instance is the fix.

```sh
# 1. Bring up the schema on the new instance.
export TARGET='postgresql://gavel:PASSWORD@/gavel?host=/cloudsql/PROJECT:REGION:INSTANCE'
DATABASE_URL="$TARGET" python initialize.py
python migrate.py upgrade --database-url "$TARGET"

# 2. Bring the old database up to the same schema, so the dump matches.
python migrate.py upgrade --database-url postgresql://gavel@localhost/gavel

# 3. Copy the data. --data-only because step 1 already built the schema.
pg_dump --data-only --no-owner --no-privileges \
        postgresql://gavel@localhost/gavel > gavel-data.sql
psql "$TARGET" < gavel-data.sql

# 4. Confirm the two agree, then repoint DATABASE_URL and redeploy.
python migrate.py status --database-url "$TARGET"
```

Migrating the old database first (step 2) matters: dumping a pre-migration
database into a migrated one fails on the columns that only exist on one side.

Re-running any of this is safe. `migrate.py upgrade` on an up-to-date database
prints `Nothing to do`.

## What the migrations do

| Migration | Effect |
|---|---|
| `0001_notes_are_unbounded` | Makes `decision.notes`, `skip.reason` and `skip.note` unbounded `TEXT`, and adds `decision.notes_item_id` so a note records which project it is about rather than being guessed at from the winner/loser pair. |
| `0002_hackathon_tenancy` | Adds the `hackathon` table and a `hackathon_id` to `item`, `annotator`, `decision`, `skip` and `setting`. Enforces one active hackathon with a partial unique index. |
| `0003_backfill_legacy_hackathon` | Adopts every pre-tenancy row into one inactive `legacy` hackathon, so past results stay exportable instead of disappearing from every scoped query. Override the id and name with `LEGACY_HACKATHON_ID` / `LEGACY_HACKATHON_NAME`. Only touches rows with a NULL `hackathon_id`, so it is safe to re-run and a no-op on a fresh instance. |
| `0004_tenancy_constraints` | Makes `hackathon_id` `NOT NULL` and adds unique indexes on `(email, hackathon_id)` and `(name, hackathon_id)`. Reports duplicates and skips the index rather than failing, so you can deduplicate and re-run. |

## Hackathons as the tenant

Gavel had no concept of an event, so one database accumulated every hackathon
forever: last year's projects stayed active and judgeable, judges kept their
crowd-BT priors and their seen/skipped history across events, and "judging is
closed" was a single global flag.

Everything is now scoped to the hackathon that is active, which Gavel takes from
`GET /hackathons/active` on the HackPSU API. Activating a new one is the whole
reset: judges get a fresh row with fresh priors and no memory of what they have
seen, only that event's projects are offered, and closing judging for one event
does not close it for the next. Previous events stay in the database, out of the
way, and can be reactivated from the admin page.

Admins set the active hackathon at `/admin/` — "Sync" adopts whatever the API
reports, and any hackathon Gavel already knows about can be reactivated
directly. With none active, Gavel says so plainly instead of rendering an empty
system that looks like the data vanished.
