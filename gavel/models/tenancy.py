"""
Resolution of the currently active hackathon.

Everything judging-related is scoped to one event. This module answers "which
event is running right now?" once per request and caches the answer on Flask's
request context, so the dozens of scoped queries a single page issues don't
each hit the database for it.

Kept separate from gavel.models.hackathon to avoid an import cycle: the models
that carry a hackathon_id need this resolver, and the resolver needs the
Hackathon model.
"""

from flask import g, has_request_context

_CACHE_KEY = '_gavel_current_hackathon_id'


class NoActiveHackathon(Exception):
    """
    Raised when judging data is touched with no active event.

    Failing loudly beats the alternative: with no tenant, scoped writes would
    land with a NULL hackathon_id and scoped reads would quietly return
    nothing, which looks like "all the projects disappeared" rather than like a
    configuration problem.
    """


def current_hackathon():
    """The active Hackathon row, or None."""
    from gavel.models.hackathon import Hackathon
    return Hackathon.current()


def current_hackathon_id(required=True):
    """
    Id of the active hackathon.

    Cached per request. Pass required=False where a missing hackathon should
    read as "no data" rather than an error -- the admin page uses that to
    render a setup prompt instead of a stack trace.
    """
    if has_request_context() and _CACHE_KEY in g:
        cached = g.get(_CACHE_KEY)
        if cached is None and required:
            raise NoActiveHackathon(
                'No active hackathon. Sync one from the HackPSU API, or set '
                'HACKATHON_ID.'
            )
        return cached

    hackathon = current_hackathon()
    hackathon_id = hackathon.id if hackathon else None

    if has_request_context():
        setattr(g, _CACHE_KEY, hackathon_id)

    if hackathon_id is None and required:
        raise NoActiveHackathon(
            'No active hackathon. Sync one from the HackPSU API, or set '
            'HACKATHON_ID.'
        )
    return hackathon_id


def forget_current_hackathon():
    """Drop the per-request cache after the active hackathon changes."""
    if has_request_context() and _CACHE_KEY in g:
        g.pop(_CACHE_KEY, None)
