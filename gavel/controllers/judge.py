from gavel import app
from gavel.models import *
from gavel.constants import *
import gavel.settings as settings
import gavel.utils as utils
import gavel.crowd_bt as crowd_bt
from gavel.firebase_session_auth import hackpsu_auth_required
from flask import (
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from numpy.random import choice, random, shuffle
from functools import wraps
from datetime import datetime

def requires_open(redirect_to):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if Setting.value_of(SETTING_CLOSED) == SETTING_TRUE:
                return redirect(url_for(redirect_to))
            else:
                return f(*args, **kwargs)
        return decorated
    return decorator


@app.route('/health')
def health():
    """Health check endpoint - no auth required"""
    return {'status': 'ok', 'service': 'gavel'}, 200

@app.route('/')
@hackpsu_auth_required
def index():
    annotator = get_current_annotator()

    # db.session.expire_all()

    if Setting.value_of(SETTING_CLOSED) == SETTING_TRUE:
        return render_template(
            'closed.html',
            content=utils.render_markdown(settings.CLOSED_MESSAGE)
        )
    if not annotator.active:
        return render_template(
            'disabled.html',
            content=utils.render_markdown(settings.DISABLED_MESSAGE)
        )
    if not annotator.read_welcome:
        return redirect(url_for('welcome'))
    maybe_init_annotator()
    if annotator.next is None:
        return render_template(
            'wait.html',
            content=utils.render_markdown(settings.WAIT_MESSAGE)
        )
    elif annotator.prev is None:
        return render_template('begin.html', item=annotator.next, timer_duration=settings.TIMER_DURATION)
    else:
        return render_template('vote.html', prev=annotator.prev, next=annotator.next, timer_duration=settings.TIMER_DURATION)

def read_notes():
    '''
    Read the judge's free-text notes from the submitted form.

    No length check: notes are stored in an unbounded TEXT column. The original
    bug was a bounded VARCHAR rejecting long notes, and capping the input would
    have been the same bug with a friendlier error message.
    '''
    return (request.form.get('notes', '').strip() or None)

def read_skip_reason():
    '''
    Read and validate the skip reason and its accompanying note.

    Returns (reason, note_or_None, error_message_or_None). The only rules here
    are that the reason is one we offer, and that "other" says what it means --
    both real constraints, unlike a length.
    '''
    reason = request.form.get('skip_reason', '').strip()
    if not reason:
        return None, None, 'Please choose a reason for skipping this project.'
    if reason not in SKIP_REASONS:
        return None, None, 'That is not a valid skip reason.'

    note = request.form.get('skip_note', '').strip()
    if reason in SKIP_REASONS_REQUIRING_NOTE and not note:
        return None, None, (
            'Please say briefly why you are skipping this project.'
        )
    return reason, (note or None), None

@app.route('/vote', methods=['POST'])
@requires_open(redirect_to='index')
@hackpsu_auth_required
def vote():
    annotator = get_current_annotator()
    if annotator is None or annotator.prev is None or annotator.next is None:
        # Stale form, double submit, or a reassignment between render and
        # submit. Sending them back to index re-renders the current state
        # instead of raising AttributeError on `annotator.prev.id`.
        return redirect(url_for('index'))

    action = request.form.get('action', '')

    # Validate everything *before* opening the transaction: with_retries can
    # call tx() more than once, and a `return` from inside it is discarded, so
    # error responses raised in there never reached the judge.
    notes = read_notes()

    skip_reason = skip_note = None
    if action == 'Skip':
        skip_reason, skip_note, error = read_skip_reason()
        if error:
            return utils.user_error(error)
        # A judge who writes notes and then skips used to lose them entirely.
        # Keep them on the Skip record, in full.
        if notes:
            skip_note = '\n\n'.join(filter(None, [skip_note, notes]))
    elif action not in ('Previous', 'Current'):
        return utils.user_error('Please choose Previous, Current, or Skip.')

    def tx():
        annotator = get_current_annotator()
        if annotator.prev is None or annotator.next is None:
            return
        if annotator.prev.id == int(request.form['prev_id']) and annotator.next.id == int(request.form['next_id']):
            if action == 'Skip':
                # Record the skip with reason
                skip = Skip(annotator, annotator.next, skip_reason, note=skip_note)
                db.session.add(skip)
                annotator.ignore.append(annotator.next)
            else:
                # The note is about the project the judge was just looking at,
                # which is `next` regardless of which one they picked.
                notes_item = annotator.next
                # ignore things that were deactivated in the middle of judging
                if annotator.prev.active and annotator.next.active:
                    if action == 'Previous':
                        perform_vote(annotator, next_won=False)
                        decision = Decision(annotator, winner=annotator.prev, loser=annotator.next, notes=notes, notes_item=notes_item)
                    else:
                        perform_vote(annotator, next_won=True)
                        decision = Decision(annotator, winner=annotator.next, loser=annotator.prev, notes=notes, notes_item=notes_item)
                    db.session.add(decision)
                elif notes:
                    # No comparison is recorded when a project was deactivated
                    # mid-judging, but the judge still wrote something about a
                    # real project. Don't throw it away.
                    db.session.add(Skip(annotator, notes_item, 'deactivated', note=notes))
                annotator.next.viewed.append(annotator) # counted as viewed even if deactivated
                annotator.prev = annotator.next
                annotator.ignore.append(annotator.prev)
            annotator.update_next(choose_next(annotator))
            db.session.commit()
    with_retries(tx)
    return redirect(url_for('index'))

@app.route('/begin', methods=['POST'])
@requires_open(redirect_to='index')
@hackpsu_auth_required
def begin():
    annotator = get_current_annotator()
    if annotator is None or annotator.next is None:
        return redirect(url_for('index'))

    action = request.form.get('action', '')
    if action not in ('Continue', 'Skip'):
        return utils.user_error('Please choose Continue or Skip.')

    # Validated up front, outside the retried transaction -- see vote().
    skip_reason = skip_note = None
    if action == 'Skip':
        skip_reason, skip_note, error = read_skip_reason()
        if error:
            return utils.user_error(error)

    def tx():
        annotator = get_current_annotator()
        if annotator.next is None:
            return
        if annotator.next.id == int(request.form['item_id']):
            annotator.ignore.append(annotator.next)
            if action == 'Continue':
                annotator.next.viewed.append(annotator)
                annotator.prev = annotator.next
                annotator.update_next(choose_next(annotator))
            else:
                # Record the skip with reason
                skip = Skip(annotator, annotator.next, skip_reason, note=skip_note)
                db.session.add(skip)
                annotator.next = None # will be reset in index
            db.session.commit()
    with_retries(tx)
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    import os
    session.pop(ANNOTATOR_ID, None)
    # Redirect to HackPSU auth server logout
    auth_logout_url = os.environ.get('AUTH_LOGOUT_URL', 'http://localhost:3000/api/sessionLogout')
    gavel_url = os.environ.get('GAVEL_URL', 'http://localhost:5000')
    return redirect(f'{auth_logout_url}?redirect={gavel_url}')

# Old secret-based login removed - using HackPSU Firebase auth instead
# @app.route('/login/<secret>/')
# def login(secret):
#     annotator = Annotator.by_secret(secret)
#     if annotator is None:
#         session.pop(ANNOTATOR_ID, None)
#         session.modified = True
#     else:
#         session[ANNOTATOR_ID] = annotator.id
#     return redirect(url_for('index'))

@app.route('/welcome/')
@requires_open(redirect_to='index')
@hackpsu_auth_required
def welcome():
    return render_template(
        'welcome.html',
        content=utils.render_markdown(settings.WELCOME_MESSAGE)
    )

@app.route('/welcome/done', methods=['POST'])
@requires_open(redirect_to='index')
@hackpsu_auth_required
def welcome_done():
    def tx():
        annotator = get_current_annotator()
        if request.form['action'] == 'Continue':
            annotator.read_welcome = True
        db.session.commit()
    with_retries(tx)
    return redirect(url_for('index'))

def get_current_annotator():
    return Annotator.by_id(session.get(ANNOTATOR_ID, None))

def preferred_items(annotator):
    '''
    Return a list of preferred items for the given annotator to look at next.

    This method uses a variety of strategies to try to select good candidate
    projects.
    '''
    items = []
    ignored_ids = {i.id for i in annotator.ignore}

    if ignored_ids:
        available_items = Item.query_current().filter(
            (Item.active == True) & (~Item.id.in_(ignored_ids))
        ).all()
    else:
        available_items = Item.query_current().filter(Item.active == True).all()

    prioritized_items = [i for i in available_items if i.prioritized]

    items = prioritized_items if prioritized_items else available_items

    annotators = Annotator.query_current().filter(
        (Annotator.active == True) & (Annotator.next != None) & (Annotator.updated != None)
    ).all()
    busy = {i.next.id for i in annotators if \
        (datetime.utcnow() - i.updated).total_seconds() < settings.TIMEOUT * 60}
    nonbusy = [i for i in items if i.id not in busy]
    preferred = nonbusy if nonbusy else items

    less_seen = [i for i in preferred if len(i.viewed) < settings.MIN_VIEWS]

    return less_seen if less_seen else preferred

def maybe_init_annotator():
    def tx():
        annotator = get_current_annotator()
        if annotator.next is None:
            items = preferred_items(annotator)
            if items:
                annotator.update_next(choice(items))
                db.session.commit()
    with_retries(tx)

def choose_next(annotator):
    items = preferred_items(annotator)

    shuffle(items) # useful for argmax case as well in the case of ties
    if items:
        if random() < crowd_bt.EPSILON:
            return items[0]
        else:
            return crowd_bt.argmax(lambda i: crowd_bt.expected_information_gain(
                annotator.alpha,
                annotator.beta,
                annotator.prev.mu,
                annotator.prev.sigma_sq,
                i.mu,
                i.sigma_sq), items)
    else:
        return None

def perform_vote(annotator, next_won):
    if next_won:
        winner = annotator.next
        loser = annotator.prev
    else:
        winner = annotator.prev
        loser = annotator.next
    u_alpha, u_beta, u_winner_mu, u_winner_sigma_sq, u_loser_mu, u_loser_sigma_sq = crowd_bt.update(
        annotator.alpha,
        annotator.beta,
        winner.mu,
        winner.sigma_sq,
        loser.mu,
        loser.sigma_sq
    )
    annotator.alpha = u_alpha
    annotator.beta = u_beta
    winner.mu = u_winner_mu
    winner.sigma_sq = u_winner_sigma_sq
    loser.mu = u_loser_mu
    loser.sigma_sq = u_loser_sigma_sq

from flask import jsonify
from sqlalchemy.orm import joinedload

def _note_records():
    """
    Every stored note, paired with the project it is actually about.

    Notes live on two tables: Decision.notes (the judge compared two projects
    and commented) and Skip.note (the judge skipped, or explained a skip).
    Both are about a single project, so both are read here.
    """
    records = []

    hackathon_id = current_hackathon_id()
    decisions = Decision.query.options(
        joinedload(Decision.winner),
        joinedload(Decision.loser),
        joinedload(Decision.notes_item),
    ).filter(
        (Decision.notes.isnot(None)) & (Decision.hackathon_id == hackathon_id)
    ).all()
    for d in decisions:
        text = (d.notes or '').strip()
        if not text:
            continue
        item = d.note_subject
        records.append({
            'annotator_id': d.annotator_id,
            'item_id': item.id if item else None,
            'project': item.name if item else '(Unknown Project)',
            'note': text,
            'time': d.time,
        })

    skips = Skip.query.options(joinedload(Skip.item)).filter(
        (Skip.note.isnot(None)) & (Skip.hackathon_id == hackathon_id)
    ).all()
    for sk in skips:
        text = (sk.note or '').strip()
        if not text:
            continue
        records.append({
            'annotator_id': sk.annotator_id,
            'item_id': sk.item_id,
            'project': sk.item.name if sk.item else '(Unknown Project)',
            'note': text,
            'time': sk.time,
        })

    records.sort(key=lambda r: r['time'], reverse=True)
    return records

@app.route('/api/judge_notes')
@hackpsu_auth_required
def get_judge_notes():
    annotator = get_current_annotator()
    if annotator is None:
        return jsonify([])

    # Newest first, as this endpoint has always claimed to be.
    notes = [{
        'project': r['project'],
        'note': r['note'],
        'time': r['time'].strftime('%Y-%m-%d %H:%M:%S'),
    } for r in _note_records() if r['annotator_id'] == annotator.id]

    return jsonify(notes)


@app.route('/api/all_notes')
@hackpsu_auth_required
def all_notes():
    """Return every project that has notes, with all judges' notes on it."""
    by_project = {}
    for r in _note_records():
        # Group by item id, not by name: two projects can share a name, and
        # grouping by name silently merges them. Identical notes from
        # different judges are both kept -- the old code de-duplicated by
        # text, so a second judge writing "Great demo" was discarded.
        key = (r['item_id'], r['project'])
        by_project.setdefault(key, []).append(r['note'])

    formatted = [
        {'project': name, 'notes': notes}
        for (_item_id, name), notes in by_project.items()
    ]
    formatted.sort(key=lambda x: x['project'].lower())

    return jsonify(formatted)


#automatic redirect when judging closes
@app.route('/api/status')
@hackpsu_auth_required
def status():
    """Returns whether judging is closed."""
    is_closed = Setting.value_of(SETTING_CLOSED) == SETTING_TRUE
    return jsonify({"closed": is_closed})

@app.route('/api/assignment_status')
@hackpsu_auth_required
def assignment_status():
    """Returns whether the current judge has a new assignment available."""
    annotator = get_current_annotator()

    # Try to assign a new item if the annotator doesn't have one
    # This mirrors the logic in the index() route
    if annotator.next is None:
        maybe_init_annotator()
        # Refresh the annotator to get the updated state
        db.session.refresh(annotator)

    has_assignment = annotator.next is not None
    return jsonify({"has_assignment": has_assignment})

