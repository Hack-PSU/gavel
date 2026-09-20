from gavel.models import db
from gavel.models.tenancy import current_hackathon_id
from gavel.models.types import hackathon_id_column, UnboundedText, PreciseDateTime
from datetime import datetime

class Decision(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    hackathon_id = db.Column(hackathon_id_column(), db.ForeignKey('hackathon.id'), nullable=False)
    annotator_id = db.Column(db.Integer, db.ForeignKey('annotator.id'))
    annotator = db.relationship('Annotator', foreign_keys=[annotator_id], uselist=False)
    winner_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    winner = db.relationship('Item', foreign_keys=[winner_id], uselist=False)
    loser_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    loser = db.relationship('Item', foreign_keys=[loser_id], uselist=False)
    time = db.Column(PreciseDateTime, default=datetime.utcnow, nullable=False)
    # LONGTEXT on MySQL: its TEXT caps at 64KB, and an unbounded notes
    # column is the entire point of the fix.
    notes = db.Column(UnboundedText, nullable=True)
    # The project the note is about -- always the one the judge was looking at,
    # which may be either the winner or the loser of this pair. Without it a
    # note can only be guessed at from the pair, which files roughly half of
    # them against the wrong project.
    notes_item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    notes_item = db.relationship('Item', foreign_keys=[notes_item_id], uselist=False)

    def __init__(self, annotator, winner, loser, notes=None, notes_item=None):
        self.hackathon_id = current_hackathon_id()
        self.annotator = annotator
        self.winner = winner
        self.loser = loser
        self.notes = notes
        self.notes_item = notes_item

    @property
    def note_subject(self):
        '''
        The item a note should be filed under.

        Rows written before `notes_item_id` existed have no subject recorded;
        fall back to the winner so legacy notes keep the placement they
        previously had rather than disappearing.
        '''
        return self.notes_item or self.winner
