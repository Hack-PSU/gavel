from gavel.models import db
from gavel.models.tenancy import current_hackathon_id
from gavel.models.types import hackathon_id_column, UnboundedText, PreciseDateTime
from datetime import datetime

class Skip(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    hackathon_id = db.Column(hackathon_id_column(), db.ForeignKey('hackathon.id'), nullable=False)
    annotator_id = db.Column(db.Integer, db.ForeignKey('annotator.id'))
    annotator = db.relationship('Annotator', foreign_keys=[annotator_id], uselist=False)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    item = db.relationship('Item', foreign_keys=[item_id], uselist=False)
    # Text, not String(100): the reason is a short slug today, but a bounded
    # column is exactly the trap that broke Decision.notes.
    reason = db.Column(UnboundedText, nullable=False)
    # Free text accompanying the skip. Required when the reason is 'other',
    # and also where a judge's notes land when they skip rather than vote --
    # those notes used to be silently discarded.
    note = db.Column(UnboundedText, nullable=True)
    time = db.Column(PreciseDateTime, default=datetime.utcnow, nullable=False)

    def __init__(self, annotator, item, reason, note=None):
        self.hackathon_id = current_hackathon_id()
        self.annotator = annotator
        self.item = item
        self.reason = reason
        self.note = note
