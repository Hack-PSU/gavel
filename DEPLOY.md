# Deploying Gavel to Cloud Run

The database is external (Cloud SQL MySQL), so the container is a single
gunicorn process serving on `$PORT`. It holds no state.

## Build, then promote

Two workflows, deliberately separate:

| Branch | Workflow | Effect |
|---|---|---|
| `master` | `publish-to-artifact-registry.yml` | builds and publishes the image |
| `production` | `deploy-production.yml` | builds, publishes **and deploys** |

Merging to master does not put anything in front of judges. Promote when ready:

```sh
git push origin master:production
```

Judging is time-critical and master takes dependency bumps, so "every merge
goes live" is the wrong default. This mirrors how apiv3 separates its
`production` branch from `main`.

The deploy tags images by commit SHA rather than `:latest`, so a running
revision always traces back to its source, and it gates on `/health` reporting
`database: ok` -- a revision that starts but cannot reach Cloud SQL fails the
workflow instead of quietly serving errors.

CI authenticates as `gavel-github-action@`, which holds only
`artifactregistry.writer` on the gavel repository, `run.developer`, and
`serviceAccountUser` on `gavel-cloud-run@` so it can deploy as the runtime
identity.

To build by hand:

```sh
docker build -f Dockerfile.prod -t gavel .
docker run -p 8080:8080 \
  -e SECRET_KEY=... \
  -e DATABASE_URL='mysql+pymysql://user:pass@host/gavel' \
  gavel
```

## Deploy

```sh
gcloud run deploy gavel \
  --image us-east4-docker.pkg.dev/hackpsu-408118/gavel/image:latest \
  --region us-east4 \
  --add-cloudsql-instances hackpsu-408118:us-east4:hackpsudb \
  --set-env-vars "DATABASE_URL=mysql+pymysql://USER:PASS@/gavel?unix_socket=/cloudsql/hackpsu-408118:us-east4:hackpsudb" \
  --set-secrets "SECRET_KEY=gavel-secret-key:latest" \
  --max-instances 8 \
  --allow-unauthenticated
```

Cloud Run mounts the Cloud SQL socket at `/cloudsql/<connection name>`;
`--add-cloudsql-instances` is what puts it there. No proxy sidecar is needed.

Prefer `--set-secrets` over `--set-env-vars` for `SECRET_KEY`: env vars are
visible to anyone with `run.services.get`.

## Connection budget

The instance allows **280 connections in total** and is shared with apiv3, so
this is a real constraint rather than a formality.

Each gunicorn worker keeps its own SQLAlchemy pool, so the ceiling is

    max-instances x workers x (DB_POOL_SIZE + DB_MAX_OVERFLOW)

At the defaults (2 workers, pool 5, overflow 2) that is **14 per instance**, so
`--max-instances 8` tops out at 112. Raising `--max-instances`, `WEB_CONCURRENCY`
or the pool settings all multiply into that number.

For contrast, the previous image ran `gunicorn -w 16` with SQLAlchemy's
defaults: up to 240 connections from one container, before Cloud Run scaled at
all.

Concurrency comes from threads instead (`WEB_THREADS`, default 8), which suits
an app whose work is database round trips.

## Schema changes

Never applied on startup. Run them deliberately, against the same image:

```sh
docker run --rm -e DATABASE_URL="$DATABASE_URL" \
  us-east4-docker.pkg.dev/hackpsu-408118/gavel/image:latest \
  python migrate.py status
```

See `MIGRATION.md`.

## Project sync

Sync happens on demand, driven by the database rather than by a clock.

When someone opens the judging or admin page, Gavel compares a
`last_project_sync` timestamp -- stored per hackathon in the `setting` table --
against `PROJECT_SYNC_INTERVAL` (default 300s). If it has gone stale, that
request refreshes the project list from the HackPSU API.

This suits Cloud Run: the database outlives any instance, so it does not matter
how often containers start and stop, and nothing needs CPU while idle. It also
means the data is only ever refreshed when someone is actually using the tool,
which is exactly when it matters.

The staleness check is a compare-and-swap on the stored timestamp, so of a
hundred judges opening the page the instant the interval lapses, exactly one
syncs and the rest read a single indexed row and move on. Verified on both
PostgreSQL and MySQL with 24 concurrent callers: one winner.

A failed sync never surfaces to the judge -- the page renders from whatever is
already in the database.

- `PROJECT_SYNC_INTERVAL` -- seconds before the list is considered stale.
- `LAZY_PROJECT_SYNC=false` -- turn the on-demand check off entirely.
- `ENABLE_PROJECT_SYNC=true` -- additionally run the old in-process
  APScheduler. Off by default, and unnecessary now: it needs
  `--min-instances 1 --no-cpu-throttling` to fire reliably.

The admin page also has a **Sync** button for both the project list and the
active hackathon, for when you do not want to wait out the interval.

## Email

`DISABLE_EMAIL=true`. Judge invites go through Celery
(`utils.send_emails.delay`), which needs a Redis broker and a separate worker
process -- neither of which exists on Cloud Run. If invites are ever needed,
use the SendGrid path (`USE_SENDGRID=true`, `SENDGRID_API_KEY`), which sends
synchronously in the request.

## What changed from the old image

| | Before | Now |
|---|---|---|
| Database | PostgreSQL inside the container, on a `VOLUME` Cloud Run discards | External Cloud SQL |
| Processes | supervisord running postgres + gunicorn | gunicorn as PID 1 |
| Port | hardcoded 5000 | `$PORT` |
| Workers | 16 | 2 x 8 threads |
| Assets | compiled on first request | compiled at build time |
| User | root | `gavel` |
| Image | postgres + supervisor + build toolchain | multi-stage, runtime only |

The old image could not keep data on Cloud Run at all: its filesystem is
in-memory and discarded when an instance goes away, so every cold start began
with an empty database.
