"""
Automatic project synchronization from HackPSU API
Runs periodically to keep Gavel projects in sync with production
"""

import requests
import os
from urllib.parse import urljoin
import time

from sqlalchemy.exc import IntegrityError

from gavel.constants import SETTING_LAST_PROJECT_SYNC
from gavel.models import (
    Hackathon,
    Item,
    Setting,
    current_hackathon_id,
    db,
    with_retries,
)
from gavel import app

HACKPSU_API_URL = os.environ.get('HACKPSU_API_URL', 'https://apiv3.hackpsu.org/judging/projects')
CATEGORY_FILTER = os.environ.get('CATEGORY_FILTER')  # Optional category filter

# Which endpoint tells us the active hackathon.
#
# /hackathons/active is declared @Roles(Role.NONE), which still requires a
# valid credential. /hackathons/active/static carries no @Roles decorator at
# all, so it is public -- the same reason /judging/projects works today without
# one. It returns the same id/name/active fields (plus an events and sponsors
# graph we ignore), so Gavel reads it and needs no credential of its own.
#
# HACKPSU_API_KEY stays optional: when set it is sent on every call, which lets
# this keep working if /active/static ever gains a @Roles decorator.
HACKPSU_API_KEY = os.environ.get('HACKPSU_API_KEY')
HACKATHON_ID_OVERRIDE = os.environ.get('HACKATHON_ID')

# Refresh on demand when someone opens the tool, rather than on a schedule.
LAZY_SYNC_ENABLED = os.environ.get('LAZY_PROJECT_SYNC', 'true').lower() == 'true'


def _api_headers():
    return {'x-api-key': HACKPSU_API_KEY} if HACKPSU_API_KEY else {}


def _active_hackathon_url():
    """Derive the active-hackathon endpoint from the configured projects URL."""
    explicit = os.environ.get('HACKPSU_HACKATHON_URL')
    if explicit:
        return explicit
    base = HACKPSU_API_URL.split('/judging/')[0]
    return urljoin(base + '/', 'hackathons/active/static')


def sync_active_hackathon():
    """
    Point Gavel at whichever hackathon the HackPSU API says is running.

    This is the switch that makes everything else reset: projects, judges,
    decisions and the "judging is closed" flag are all scoped to the active
    hackathon, so activating a new one gives a clean event without anyone
    clearing the database.
    """
    if HACKATHON_ID_OVERRIDE:
        # Escape hatch for local development and for running a past event.
        with app.app_context():
            def tx():
                Hackathon.activate(HACKATHON_ID_OVERRIDE,
                                   'Hackathon %s' % HACKATHON_ID_OVERRIDE)
                db.session.commit()
            with_retries(tx)
            print('[SYNC] Active hackathon pinned to %s by HACKATHON_ID'
                  % HACKATHON_ID_OVERRIDE)
        return HACKATHON_ID_OVERRIDE

    url = _active_hackathon_url()
    try:
        response = requests.get(url, headers=_api_headers(), timeout=10)
    except requests.RequestException as e:
        print('[SYNC ERROR] Could not reach %s: %s' % (url, e))
        return None

    if response.status_code != 200:
        print('[SYNC ERROR] %s returned status %s%s' % (
            url, response.status_code,
            ' (endpoint now needs a credential -- set HACKPSU_API_KEY)'
            if response.status_code in (401, 403) else ''))
        return None

    data = response.json() or {}
    hackathon_id = data.get('id')
    if not hackathon_id:
        print('[SYNC ERROR] No active hackathon in the API response')
        return None

    with app.app_context():
        def tx():
            Hackathon.activate(hackathon_id, data.get('name') or hackathon_id)
            db.session.commit()
        with_retries(tx)
        print('[SYNC] Active hackathon: %s (%s)'
              % (data.get('name') or hackathon_id, hackathon_id))
    return hackathon_id

def extract_table_number(name):
    """Extract table number from project name like '(1) Space Goggles'"""
    import re
    match = re.match(r'\((\d+)\)\s*(.*)', name)
    if match:
        table_num = match.group(1)
        clean_name = match.group(2).strip()
        return table_num, clean_name
    return None, name

def matches_category_filter(categories_str, filter_category):
    """
    Check if any category in the comma-separated categories string matches the filter.

    Args:
        categories_str: Comma-separated string of categories (e.g., "Web,Mobile,AI")
        filter_category: The category to filter for (e.g., "Web")

    Returns:
        True if any category matches the filter, False otherwise
    """
    if not categories_str:
        return False

    # Split by comma and strip whitespace from each category
    categories = [cat.strip() for cat in categories_str.split(',')]
    print(categories)

    # Check if the filter category matches any of the project's categories
    return filter_category in categories

def sync_projects_from_api():
    """Fetch projects from HackPSU API and sync to Gavel"""
    print("[SYNC] Starting project sync from HackPSU API...")

    if CATEGORY_FILTER:
        print(f"[SYNC] Filtering projects by category: {CATEGORY_FILTER}")

    hackathon_id = sync_active_hackathon()
    if not hackathon_id:
        print('[SYNC] No active hackathon; skipping project sync')
        return

    try:
        # Fetch projects from API
        response = requests.get(HACKPSU_API_URL, headers=_api_headers(),
                                timeout=10)

        if response.status_code != 200:
            print(f"[SYNC ERROR] API returned status {response.status_code}")
            return

        projects = response.json()
        print(f"[SYNC] Fetched {len(projects)} projects from API")

        with app.app_context():
            synced_count = 0
            updated_count = 0
            filtered_count = 0

            # Pre-filter projects outside the transaction
            projects_to_sync = []
            for project_data in projects:
                if CATEGORY_FILTER:
                    categories = project_data.get('categories', '')
                    if not matches_category_filter(categories, CATEGORY_FILTER):
                        filtered_count += 1
                        continue
                projects_to_sync.append(project_data)

            def tx():
                nonlocal synced_count, updated_count
                synced_count = 0
                updated_count = 0

                for project_data in projects_to_sync:
                    project_id = project_data.get('id')
                    raw_name = project_data.get('name', f'Project {project_id}')

                    table_num, clean_name = extract_table_number(raw_name)

                    if table_num:
                        location = f"Table {table_num}"
                    else:
                        # Falling back to the database id printed a row id as
                        # if it were a physical table number, sending judges to
                        # a table that doesn't exist. Say it's unassigned so an
                        # admin can see and fix it.
                        location = "Table not assigned"
                        print(f"[SYNC] No table number in name: {raw_name!r}")

                    description = clean_name

                    # Scoped to this hackathon: a project with the same name
                    # at a previous event is a different project.
                    existing = Item.query.filter_by(
                        name=clean_name, hackathon_id=hackathon_id).first()

                    if not existing:
                        item = Item(
                            name=clean_name,
                            location=location,
                            description=description,
                            hackathon_id=hackathon_id
                        )
                        item.active = True
                        db.session.add(item)
                        synced_count += 1
                        print(f"[SYNC] Created: {clean_name} at {location}")
                    else:
                        if existing.location != location:
                            existing.location = location
                            updated_count += 1
                            print(f"[SYNC] Updated location for: {clean_name} to {location}")

                db.session.commit()

            with_retries(tx)

            if CATEGORY_FILTER:
                print(f"[SYNC] Complete: {synced_count} created, {updated_count} updated, {filtered_count} filtered out")
            else:
                print(f"[SYNC] Complete: {synced_count} created, {updated_count} updated")

    except Exception as e:
        print(f"[SYNC ERROR] Failed to sync projects: {e}")
        import traceback
        traceback.print_exc()

# Arbitrary but fixed key for the PostgreSQL advisory lock that elects the one
# process allowed to run the sync scheduler.
SYNC_LOCK_KEY = 0x6761766C  # b'gavl' -- PostgreSQL advisory locks are numeric
SYNC_LOCK_NAME = 'gavel_project_sync'  # MySQL's GET_LOCK takes a name

# The connection holding the advisory lock. Must stay open for the lifetime of
# the process -- PostgreSQL releases a session-level advisory lock when the
# session ends, which is exactly the behaviour we want if a worker dies.
_sync_lock_conn = None


def _claim_sync_leadership():
    """
    Try to become the single process responsible for project sync.

    setup_project_sync() runs at import time, so with `gunicorn -w 16` all
    sixteen workers previously started their own APScheduler, each firing a
    full sync every PROJECT_SYNC_INTERVAL seconds and each racing the others to
    insert the same projects. A session-level advisory lock elects exactly one
    of them; if that worker dies the lock is released and the next boot elects
    another.
    """
    global _sync_lock_conn
    try:
        with app.app_context():
            dialect = db.engine.dialect.name
            conn = db.engine.raw_connection()
            cursor = conn.cursor()
            if dialect == 'postgresql':
                cursor.execute('SELECT pg_try_advisory_lock(%s)',
                               (SYNC_LOCK_KEY,))
            elif dialect == 'mysql':
                # GET_LOCK is MySQL's session-level advisory lock and behaves
                # like the PostgreSQL one: released when the session ends, so a
                # worker that dies hands leadership back automatically.
                cursor.execute('SELECT GET_LOCK(%s, 0)', (SYNC_LOCK_NAME,))
            else:
                cursor.close()
                conn.close()
                return True  # nothing to coordinate against
            acquired = bool(cursor.fetchone()[0])
            cursor.close()
            if acquired:
                _sync_lock_conn = conn  # keep the session (and the lock) alive
            else:
                conn.close()
            return acquired
    except Exception as e:
        # Never let lock plumbing stop the app from booting. Falling back to
        # "this worker syncs" is the old behaviour, which is safe if noisy.
        print(f"[SYNC] Could not acquire sync lock ({e}); syncing from this worker")
        return True



# ---------------------------------------------------------------- lazy sync
#
# Cloud Run scales to zero and throttles CPU between requests, so a background
# scheduler does not reliably fire. Instead the freshness of the project list
# is checked when someone actually opens the tool, and a sync runs only if it
# has gone stale. The database is the shared clock: it outlives any instance,
# so this works no matter how often Cloud Run starts and stops containers.

def _sync_due(hackathon_id, interval):
    """
    Claim the right to sync, atomically.

    Returns True for exactly one caller per interval. Without the claim, a
    hundred judges opening the page the moment the interval lapses would all
    see a stale timestamp and all start syncing.

    The claim is a compare-and-swap on the stored timestamp: the UPDATE only
    matches while the value is still the one that was read, so of two racing
    requests the second changes no rows and backs off. That needs no advisory
    lock, and behaves the same on PostgreSQL and MySQL.
    """
    now = int(time.time())
    setting = Setting.by_key(SETTING_LAST_PROJECT_SYNC, hackathon_id)

    if setting is None:
        # First sync for this hackathon. Whoever inserts the row wins; the
        # primary key makes the loser fail rather than double-sync.
        try:
            db.session.add(
                Setting(SETTING_LAST_PROJECT_SYNC, str(now), hackathon_id))
            db.session.commit()
            return True
        except IntegrityError:
            db.session.rollback()
            return False

    try:
        last = int(setting.value)
    except (TypeError, ValueError):
        last = 0
    if now - last < interval:
        return False

    previous = setting.value
    claimed = db.session.execute(
        Setting.__table__.update().where(
            (Setting.__table__.c.key == SETTING_LAST_PROJECT_SYNC) &
            (Setting.__table__.c.hackathon_id == hackathon_id) &
            (Setting.__table__.c.value == previous)
        ).values(value=str(now))
    ).rowcount
    db.session.commit()
    return claimed == 1


def maybe_sync_projects(interval=None):
    """
    Sync from the HackPSU API if the data has gone stale.

    Safe to call on any page load: the common case is a single indexed read of
    one settings row, and at most one request per interval does any real work.
    Never raises -- a judge opening the voting page must not see an error
    because the upstream API is briefly unavailable.
    """
    if not LAZY_SYNC_ENABLED:
        return False

    interval = interval or int(os.environ.get('PROJECT_SYNC_INTERVAL', 300))
    try:
        hackathon_id = current_hackathon_id(required=False)
        if hackathon_id is None:
            # No active event yet; syncing one is an explicit admin action.
            return False
        if not _sync_due(hackathon_id, interval):
            return False

        print('[SYNC] project list is stale; refreshing')
        sync_projects_from_api()
        return True
    except Exception as e:
        # The timestamp has already been claimed, so a failure here means the
        # next attempt waits out the interval rather than retrying in a loop
        # against an API that is evidently unhappy.
        print('[SYNC ERROR] lazy sync failed: %s' % e)
        try:
            db.session.rollback()
        except Exception:
            pass
        return False


def setup_project_sync():
    """Set up periodic project sync using APScheduler"""
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

    # Ensure database tables exist before syncing
    with app.app_context():
        db.create_all()

    if not _claim_sync_leadership():
        print("[SYNC] Another worker owns project sync; not scheduling here")
        return None

    # Get sync interval from env (default 5 minutes)
    sync_interval = int(os.environ.get('PROJECT_SYNC_INTERVAL', 300))

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=sync_projects_from_api,
        trigger="interval",
        seconds=sync_interval,
        id='sync_projects',
        name='Sync projects from HackPSU API',
        replace_existing=True,
        # If a sync overruns the interval, run it once when it catches up
        # rather than queueing up a backlog of identical syncs.
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()

    # Run initial sync immediately
    print("[SYNC] Running initial project sync...")
    sync_projects_from_api()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    print(f"[SYNC] Project sync scheduled every {sync_interval} seconds")
    return scheduler
