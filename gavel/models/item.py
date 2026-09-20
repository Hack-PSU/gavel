from gavel.models import db
from gavel.models.tenancy import current_hackathon_id
from gavel.models.types import hackathon_id_column, DoubleFloat, PreciseDateTime
import gavel.crowd_bt as crowd_bt
from sqlalchemy.orm.exc import NoResultFound

view_table = db.Table('view',
    db.Column('item_id', db.Integer, db.ForeignKey('item.id')),
    db.Column('annotator_id', db.Integer, db.ForeignKey('annotator.id'))
)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    # The event this project belongs to. Without it, last year's projects stay
    # active and judges get sent to tables that no longer exist.
    hackathon_id = db.Column(hackathon_id_column(), db.ForeignKey('hackathon.id'), nullable=False)
    name = db.Column(db.Text, nullable=False)
    location = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=False)
    active = db.Column(db.Boolean, default=True, nullable=False)
    viewed = db.relationship('Annotator', secondary=view_table)
    prioritized = db.Column(db.Boolean, default=False, nullable=False)

    mu = db.Column(DoubleFloat)
    sigma_sq = db.Column(DoubleFloat)

    def __init__(self, name, location, description, hackathon_id=None):
        self.hackathon_id = hackathon_id or current_hackathon_id()
        self.name = name
        self.location = location
        self.description = description
        self.mu = crowd_bt.MU_PRIOR
        self.sigma_sq = crowd_bt.SIGMA_SQ_PRIOR

    @classmethod
    def by_id(cls, uid):
        if uid is None:
            return None
        try:
            item = cls.query.get(uid)
        except NoResultFound:
            item = None
        # An id from one event must never resolve while another is running.
        if item is not None and item.hackathon_id != current_hackathon_id():
            return None
        return item

    @classmethod
    def query_current(cls):
        '''Projects belonging to the active hackathon.'''
        return cls.query.filter(cls.hackathon_id == current_hackathon_id())
