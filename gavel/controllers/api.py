from gavel import app
from gavel.models import *
import gavel.utils as utils
from gavel.firebase_session_auth import hackpsu_admin_required
from flask import Response, jsonify
from sqlalchemy.orm import joinedload

@app.route('/api/items.csv')
@app.route('/api/projects.csv')
@hackpsu_admin_required
def item_dump():
    items = Item.query_current().order_by(desc(Item.mu)).all()
    data = [['Mu', 'Sigma Squared', 'Name', 'Location', 'Description', 'Active']]
    data += [[
        str(item.mu),
        str(item.sigma_sq),
        item.name,
        item.location,
        item.description,
        item.active
    ] for item in items]
    return Response(utils.data_to_csv_string(data), mimetype='text/csv')

@app.route('/api/annotators.csv')
@app.route('/api/judges.csv')
@hackpsu_admin_required
def annotator_dump():
    annotators = Annotator.query_current().all()
    data = [['Name', 'Email', 'Description', 'Secret']]
    data += [[
        str(a.name),
        a.email,
        a.description,
        a.secret
    ] for a in annotators]
    return Response(utils.data_to_csv_string(data), mimetype='text/csv')

@app.route('/api/decisions.json')
@hackpsu_admin_required
def decisions_json():
    decisions = Decision.query.options(
        joinedload(Decision.annotator),
        joinedload(Decision.winner),
        joinedload(Decision.loser),
    ).filter(
        Decision.hackathon_id == current_hackathon_id()
    ).order_by(Decision.time.desc()).all()
    return jsonify([{
        'id': d.id,
        'annotator_id': d.annotator_id,
        'annotator_name': d.annotator.name if d.annotator else '(deleted judge)',
        'winner_id': d.winner_id,
        'winner_name': d.winner.name if d.winner else '(deleted project)',
        'loser_id': d.loser_id,
        'loser_name': d.loser.name if d.loser else '(deleted project)',
        'time': d.time.strftime('%Y-%m-%d %H:%M:%S'),
        'notes': d.notes or '',
        'notes_project': d.note_subject.name if d.note_subject else '',
    } for d in decisions])

@app.route('/api/decisions.csv')
@hackpsu_admin_required
def decisions_dump():
    decisions = Decision.query.options(
        joinedload(Decision.annotator),
        joinedload(Decision.winner),
        joinedload(Decision.loser),
    ).filter(Decision.hackathon_id == current_hackathon_id()).all()
    data = [['Annotator ID', 'Winner ID', 'Loser ID', 'Time', 'Notes About', 'Notes']]
    data += [[
        str(d.annotator_id),
        str(d.winner_id),
        str(d.loser_id),
        str(d.time),
        str(d.notes_item_id or ''),
        d.notes or ''
    ] for d in decisions]
    return Response(utils.data_to_csv_string(data), mimetype='text/csv')
