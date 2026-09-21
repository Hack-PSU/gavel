from gavel.models import db
from gavel.models.types import hackathon_id_column, PreciseDateTime
from sqlalchemy.orm.exc import NoResultFound
from datetime import datetime


class Hackathon(db.Model):
    '''
    The tenant that owns every piece of judging data.

    Gavel had no concept of an event, so a single database accumulated every
    hackathon forever: last year's projects stayed active and judgeable, judges
    kept their crowd-BT priors and their seen/skipped history across events, and
    "judging is closed" was one global flag. Scoping everything to a hackathon
    is what makes each event start clean without anyone remembering to wipe the
    database by hand.

    `id` is the hackathon id from the HackPSU API, so the two systems agree on
    which event is running without a mapping table.
    '''
    __tablename__ = 'hackathon'

    id = db.Column(hackathon_id_column(), primary_key=True, nullable=False)
    name = db.Column(db.Text, nullable=False)
    active = db.Column(db.Boolean, default=False, nullable=False)
    # Disposable: a demo hackathon exists so judges can be trained on the real
    # system, and everything belonging to it is deleted when demo mode ends.
    # Only a hackathon carrying this flag may be torn down.
    demo = db.Column(db.Boolean, default=False, nullable=False)
    synced = db.Column(PreciseDateTime)

    def __init__(self, id, name, active=False, demo=False):
        self.id = id
        self.name = name
        self.active = active
        self.demo = demo
        self.synced = datetime.utcnow()

    @classmethod
    def current(cls):
        '''The active hackathon, or None if nothing has been synced yet.'''
        return cls.query.filter(cls.active == True).first()

    @classmethod
    def by_id(cls, hackathon_id):
        if hackathon_id is None:
            return None
        try:
            return cls.query.get(hackathon_id)
        except NoResultFound:
            return None

    @classmethod
    def upsert(cls, hackathon_id, name):
        """
        Record a hackathon without changing which one is active.

        Used when the API reports an event that must not be switched to yet --
        during a demo, the demo owns the active flag.
        """
        hackathon = cls.by_id(hackathon_id)
        if hackathon is None:
            hackathon = cls(hackathon_id, name)
            db.session.add(hackathon)
        else:
            hackathon.name = name
        hackathon.synced = datetime.utcnow()
        return hackathon

    @classmethod
    def activate(cls, hackathon_id, name):
        '''
        Make `hackathon_id` the active tenant, creating it if needed.

        Only one hackathon may be active at a time; the database enforces that
        with a partial unique index, and this keeps the rows consistent with it.
        '''
        hackathon = cls.by_id(hackathon_id)
        if hackathon is None:
            hackathon = cls(hackathon_id, name)
            db.session.add(hackathon)
        else:
            hackathon.name = name
        hackathon.synced = datetime.utcnow()

        # Stand every other hackathon down before standing this one up, so the
        # single-active index is never transiently violated.
        for other in cls.query.filter(cls.active == True).all():
            if other.id != hackathon_id:
                other.active = False
        db.session.flush()
        hackathon.active = True
        return hackathon
