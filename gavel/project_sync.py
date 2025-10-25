"""
Automatic project synchronization from HackPSU API
Runs periodically to keep Gavel projects in sync with production
"""

import requests
import os
from gavel.models import Item, db
from gavel import app

HACKPSU_API_URL = os.environ.get('HACKPSU_API_URL', 'https://apiv3.hackpsu.org/judging/projects')
CATEGORY_FILTER = os.environ.get('CATEGORY_FILTER')  # Optional category filter

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

    # Check if the filter category matches any of the project's categories
    return filter_category in categories

def sync_projects_from_api():
    """Fetch projects from HackPSU API and sync to Gavel"""
    print("[SYNC] Starting project sync from HackPSU API...")

    if CATEGORY_FILTER:
        print(f"[SYNC] Filtering projects by category: {CATEGORY_FILTER}")

    try:
        # Fetch projects from API
        response = requests.get(HACKPSU_API_URL, timeout=10)

        if response.status_code != 200:
            print(f"[SYNC ERROR] API returned status {response.status_code}")
            return

        projects = response.json()
        print(f"[SYNC] Fetched {len(projects)} projects from API")

        with app.app_context():
            synced_count = 0
            updated_count = 0
            filtered_count = 0

            for project_data in projects:
                # Apply category filter if specified
                if CATEGORY_FILTER:
                    categories = project_data.get('categories', '')
                    if not matches_category_filter(categories, CATEGORY_FILTER):
                        filtered_count += 1
                        continue


                project_id = project_data.get('id')
                raw_name = project_data.get('name', f'Project {project_id}')

                # Extract table number from name like "(1) Space Goggles"
                table_num, clean_name = extract_table_number(raw_name)

                # Use extracted table number, or fallback to project ID
                if table_num:
                    location = f"Table {table_num}"
                else:
                    location = f"Table {project_id}"

                # Description is just the clean name
                description = clean_name

                # Check if project already exists by clean name
                existing = Item.query.filter_by(name=clean_name).first()

                if not existing:
                    # Create new project
                    item = Item(
                        name=clean_name,
                        location=location,
                        description=description
                    )
                    item.active = True
                    db.session.add(item)
                    synced_count += 1
                    print(f"[SYNC] Created: {clean_name} at {location}")
                else:
                    # Update existing project location if changed
                    if existing.location != location:
                        existing.location = location
                        updated_count += 1
                        print(f"[SYNC] Updated location for: {clean_name} to {location}")

            db.session.commit()
            if CATEGORY_FILTER:
                print(f"[SYNC] Complete: {synced_count} created, {updated_count} updated, {filtered_count} filtered out")
            else:
                print(f"[SYNC] Complete: {synced_count} created, {updated_count} updated")

    except Exception as e:
        print(f"[SYNC ERROR] Failed to sync projects: {e}")
        import traceback
        traceback.print_exc()

def setup_project_sync():
    """Set up periodic project sync using APScheduler"""
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

    # Get sync interval from env (default 5 minutes)
    sync_interval = int(os.environ.get('PROJECT_SYNC_INTERVAL', 300))

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=sync_projects_from_api,
        trigger="interval",
        seconds=sync_interval,
        id='sync_projects',
        name='Sync projects from HackPSU API',
        replace_existing=True
    )
    scheduler.start()

    # Run initial sync immediately
    print("[SYNC] Running initial project sync...")
    sync_projects_from_api()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    print(f"[SYNC] Project sync scheduled every {sync_interval} seconds")
    return scheduler
