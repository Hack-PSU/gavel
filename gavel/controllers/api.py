from gavel import app
from gavel.models import *
from gavel.firebase_session_auth import hackpsu_admin_required
from flask import Response, jsonify
from sqlalchemy.orm import joinedload

@app.route('/api/items.csv')
@app.route('/api/projects.csv')
@hackpsu_admin_required
def item_dump():
    items = Item.query.order_by(desc(Item.mu)).all()
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
    annotators = Annotator.query.all()
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
    ).order_by(Decision.time.desc()).all()
    return jsonify([{
        'id': d.id,
        'annotator_id': d.annotator.id,
        'annotator_name': d.annotator.name,
        'winner_id': d.winner.id,
        'winner_name': d.winner.name,
        'loser_id': d.loser.id,
        'loser_name': d.loser.name,
        'time': d.time.strftime('%Y-%m-%d %H:%M:%S'),
        'notes': d.notes or '',
    } for d in decisions])

@app.route('/api/decisions.csv')
@hackpsu_admin_required
def decisions_dump():
    decisions = Decision.query.all()
    data = [['Annotator ID', 'Winner ID', 'Loser ID', 'Time']]
    data += [[
        str(d.annotator.id),
        str(d.winner.id),
        str(d.loser.id),
        str(d.time)
    ] for d in decisions]
    return Response(utils.data_to_csv_string(data), mimetype='text/csv')
