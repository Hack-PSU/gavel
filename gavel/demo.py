"""
Demo mode: run a real event that is thrown away afterwards.

Judge workshops need the genuine experience -- log in, get assigned a project,
walk to a table, vote, leave notes, watch the leaderboard move. Doing that on
the live event pollutes the results everyone is about to rely on.

Because every piece of judging data already belongs to a hackathon, a demo is
just another hackathon that happens to be disposable. Nothing in the voting,
analytics or admin code knows demo mode exists: judges are assigned projects,
crowd-BT updates scores, notes are stored, exactly as always. The only
difference is that ending the demo deletes that tenant.

The safety property that matters: teardown is refused unless the hackathon
carries the `demo` flag, so no sequence of clicks or mistyped ids can delete a
real event's results.
"""

from datetime import datetime

import os
import time

from gavel.constants import SETTING_DEMO_EXPIRES_AT, SETTING_DEMO_RESTORE_TO
from gavel.models import (
    Annotator,
    Decision,
    Hackathon,
    Item,
    Setting,
    Skip,
    db,
    forget_current_hackathon,
    ignore_table,
    view_table,
)

# A demo's id is derived from the real hackathon it mirrors: "demo:<real-id>".
#
# HackPSU's API owns hackathon ids and has no demo, so Gavel has to make one
# up. Deriving it keeps the relationship visible -- you can always tell which
# event a demo shadows, and ending it knows where to return -- and the colon
# guarantees no collision, since API ids are alphanumeric tokens.
DEMO_ID_PREFIX = 'demo:'


# A demo that is forgotten is worse than no demo at all: judging stays pointed
# at the throwaway tenant, so when the real event starts, real votes land in it
# -- and are deleted when someone finally clicks End Demo. Demos therefore
# expire on their own.
DEFAULT_DURATION_MINUTES = int(os.environ.get('DEMO_DURATION_MINUTES', 240))


def demo_id_for(hackathon_id):
    return DEMO_ID_PREFIX + (hackathon_id or 'standalone')


def real_id_behind(demo_hackathon_id):
    """The hackathon a demo id was derived from, or None."""
    if not demo_hackathon_id.startswith(DEMO_ID_PREFIX):
        return None
    real = demo_hackathon_id[len(DEMO_ID_PREFIX):]
    return None if real == 'standalone' else real


class DemoError(Exception):
    """Raised when a demo operation would be unsafe or makes no sense."""


def active_demo():
    """The demo hackathon currently running, or None."""
    return Hackathon.query.filter(
        (Hackathon.demo == True) & (Hackathon.active == True)
    ).first()


def start_demo(name=None, copy_projects_from=None,
               duration_minutes=None):
    """
    Begin a demo, seeded with a copy of the real projects.

    Judges are shown the same project names and table numbers they would see
    on the day, because a workshop where every project is called "Test 1"
    teaches nobody where to walk. The copies are new rows in the demo tenant
    with fresh scores, so voting on them cannot touch the real ones.

    The hackathon that was active is remembered, so ending the demo puts
    things back exactly as they were.
    """
    if active_demo():
        raise DemoError('A demo is already running.')

    previous = Hackathon.query.filter(Hackathon.active == True).first()
    source_id = copy_projects_from or (previous.id if previous else None)

    demo_id = demo_id_for(source_id)
    demo_name = name or 'Demo of %s' % (previous.name if previous else 'Gavel')

    # A demo of the same event may be run repeatedly; reuse of the id is fine
    # because the previous one was deleted when that demo ended.
    if Hackathon.by_id(demo_id) is not None:
        raise DemoError(
            'A demo tenant for %s already exists (%s). End it before starting '
            'another.' % (source_id, demo_id))

    demo = Hackathon(demo_id, demo_name, demo=True)
    db.session.add(demo)
    db.session.flush()

    copied = 0
    if source_id:
        for item in Item.query.filter(
            (Item.hackathon_id == source_id) & (Item.active == True)
        ).all():
            db.session.add(Item(
                name=item.name,
                location=item.location,
                description=item.description,
                hackathon_id=demo_id,
            ))
            copied += 1

    # Remember where to return to. Global scope: which event a demo interrupted
    # is a property of the deployment, not of either hackathon.
    Setting.set(SETTING_DEMO_RESTORE_TO, previous.id if previous else '',
                scoped=False)

    minutes = duration_minutes or DEFAULT_DURATION_MINUTES
    Setting.set(SETTING_DEMO_EXPIRES_AT,
                str(int(time.time()) + minutes * 60), scoped=False)

    Hackathon.activate(demo_id, demo_name)
    demo.demo = True
    db.session.commit()
    forget_current_hackathon()

    return demo, copied


def end_demo():
    """
    End the demo and delete everything it produced.

    Refuses to touch a hackathon that is not flagged demo. That check is the
    whole safety story, so it happens before a single row is deleted.
    """
    demo = active_demo()
    if demo is None:
        raise DemoError('No demo is running.')
    if not demo.demo:
        # Unreachable through active_demo(), but this function deletes data --
        # it re-checks rather than trusting its caller.
        raise DemoError('%s is not a demo hackathon; refusing to delete it.'
                        % demo.id)

    demo_id = demo.id
    # Prefer the event the API most recently reported; fall back to the id the
    # demo was derived from, so a demo started before any sync still restores.
    restore_to = (Setting.value_of(SETTING_DEMO_RESTORE_TO, scoped=False)
                  or real_id_behind(demo.id))

    deleted = _purge(demo_id)

    # Put the real event back before removing the demo, so the database is
    # never left with no active hackathon if something fails in between.
    restored = None
    if restore_to:
        target = Hackathon.by_id(restore_to)
        if target:
            Hackathon.activate(target.id, target.name)
            restored = target
    if restored is None:
        demo.active = False

    db.session.delete(demo)
    Setting.set(SETTING_DEMO_RESTORE_TO, '', scoped=False)
    Setting.set(SETTING_DEMO_EXPIRES_AT, '', scoped=False)
    db.session.commit()
    forget_current_hackathon()

    return demo_id, deleted, restored


def _purge(hackathon_id):
    """
    Delete every row belonging to a hackathon, children first.

    The join tables have no hackathon column of their own, so they are cleared
    via the annotators and items that own the rows. Annotators go before items
    because annotator.next_id and prev_id point at items.
    """
    counts = {}

    annotator_ids = [a.id for a in Annotator.query.filter(
        Annotator.hackathon_id == hackathon_id).all()]
    item_ids = [i.id for i in Item.query.filter(
        Item.hackathon_id == hackathon_id).all()]

    counts['decision'] = Decision.query.filter(
        Decision.hackathon_id == hackathon_id).delete(synchronize_session=False)
    counts['skip'] = Skip.query.filter(
        Skip.hackathon_id == hackathon_id).delete(synchronize_session=False)

    if item_ids:
        counts['view'] = db.session.execute(
            view_table.delete().where(view_table.c.item_id.in_(item_ids))
        ).rowcount
        counts['ignore'] = db.session.execute(
            ignore_table.delete().where(ignore_table.c.item_id.in_(item_ids))
        ).rowcount
    else:
        counts['view'] = counts['ignore'] = 0

    if annotator_ids:
        # An annotator still pointing at an item blocks the item's deletion.
        Annotator.query.filter(Annotator.id.in_(annotator_ids)).update(
            {'next_id': None, 'prev_id': None}, synchronize_session=False)
        db.session.flush()

    counts['annotator'] = Annotator.query.filter(
        Annotator.hackathon_id == hackathon_id).delete(synchronize_session=False)
    counts['item'] = Item.query.filter(
        Item.hackathon_id == hackathon_id).delete(synchronize_session=False)
    counts['setting'] = Setting.query.filter(
        Setting.hackathon_id == hackathon_id).delete(synchronize_session=False)

    db.session.flush()
    return counts


def expires_at():
    """Unix timestamp when the running demo ends itself, or None."""
    raw = Setting.value_of(SETTING_DEMO_EXPIRES_AT, scoped=False)
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


def seconds_remaining():
    deadline = expires_at()
    return None if deadline is None else max(0, deadline - int(time.time()))


def extend(minutes):
    """Push the deadline out, for a workshop that is running long."""
    if active_demo() is None:
        raise DemoError('No demo is running.')
    base = max(expires_at() or 0, int(time.time()))
    Setting.set(SETTING_DEMO_EXPIRES_AT, str(base + minutes * 60), scoped=False)
    db.session.commit()
    return seconds_remaining()


def expire_if_due():
    """
    End the demo if its time is up.

    Called on page loads, alongside the project-freshness check, so it needs no
    scheduler -- the same reason that check lives there. Returns the purge
    summary if it acted, otherwise None.

    Never raises: this runs on the judging page, and a judge must not see an
    error because a demo happened to lapse on their request.
    """
    try:
        if active_demo() is None:
            return None
        remaining = seconds_remaining()
        if remaining is None or remaining > 0:
            return None

        demo_id, deleted, restored = end_demo()
        print('[DEMO] %s expired and was deleted; active: %s'
              % (demo_id, restored.id if restored else 'none'))
        return demo_id, deleted, restored
    except DemoError:
        # Another request ended it first.
        return None
    except Exception as e:
        print('[DEMO ERROR] could not expire the demo: %s' % e)
        try:
            db.session.rollback()
        except Exception:
            pass
        return None
