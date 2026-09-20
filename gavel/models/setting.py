from gavel.models import db
from gavel.models.tenancy import current_hackathon_id
from gavel.models.types import hackathon_id_column, setting_key_column
from sqlalchemy.orm.exc import NoResultFound

# Settings that belong to Gavel as a whole rather than to one event. Everything
# else -- notably "judging is closed" -- is per-hackathon, so closing judging
# for one event doesn't close it for the next one too.
GLOBAL_SCOPE = ''


class Setting(db.Model):
    # Composite primary key: the same key exists once per hackathon. '' is the
    # global scope, because a primary key column cannot be NULL.
    # Both halves of the primary key are bounded: MySQL cannot build a
    # primary key over unbounded text.
    key = db.Column(setting_key_column(), nullable=False, primary_key=True)
    hackathon_id = db.Column(hackathon_id_column(), nullable=False, primary_key=True,
                             default=GLOBAL_SCOPE, server_default=GLOBAL_SCOPE)
    value = db.Column(db.Text, nullable=False)

    def __init__(self, key, value, hackathon_id=GLOBAL_SCOPE):
        self.key = key
        self.value = value
        self.hackathon_id = hackathon_id

    @staticmethod
    def _scope(hackathon_id, scoped):
        if not scoped:
            return GLOBAL_SCOPE
        return hackathon_id if hackathon_id is not None else current_hackathon_id()

    @classmethod
    def by_key(cls, key, hackathon_id=None, scoped=True):
        scope = cls._scope(hackathon_id, scoped)
        try:
            setting = cls.query.filter(
                (cls.key == key) & (cls.hackathon_id == scope)
            ).one()
        except NoResultFound:
            setting = None
        return setting

    @classmethod
    def value_of(cls, key, hackathon_id=None, scoped=True):
        setting = cls.by_key(key, hackathon_id, scoped)
        if setting:
            return setting.value
        else:
            return None

    @classmethod
    def set(cls, key, value, hackathon_id=None, scoped=True):
        scope = cls._scope(hackathon_id, scoped)
        setting = cls.by_key(key, scope, scoped)
        if setting:
            setting.value = value
        else:
            setting = cls(key, value, scope)
            db.session.add(setting)
