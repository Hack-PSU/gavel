import gavel.crowd_bt as crowd_bt
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy.exc
import time

class SerializableAlchemy(SQLAlchemy):
    def apply_driver_hacks(self, app, info, options):
        if not 'isolation_level' in options:
            # XXX is this slow? are there better ways?
            options['isolation_level'] = 'SERIALIZABLE'
        return super(SerializableAlchemy, self).apply_driver_hacks(app, info, options)
db = SerializableAlchemy()

from gavel.models.types import UnboundedText

# Hackathon first: every other model carries a foreign key to it.
from gavel.models.hackathon import Hackathon
from gavel.models.tenancy import (
    current_hackathon,
    current_hackathon_id,
    forget_current_hackathon,
    NoActiveHackathon,
)
from gavel.models.annotator import Annotator, ignore_table
from gavel.models.item import Item, view_table
from gavel.models.decision import Decision
from gavel.models.setting import Setting, GLOBAL_SCOPE
from gavel.models.skip import Skip

from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import desc


# How many times a transaction may be retried before giving up. The original
# loop retried forever, which is fine when the only retryable error is a
# PostgreSQL serialization failure under low contention -- but MySQL under
# SERIALIZABLE turns plain reads into locking reads, so a genuinely stuck pair
# of transactions could spin here indefinitely while holding a worker.
MAX_RETRIES = 8

# SQLSTATE 40001 is "serialization failure" and is what both engines report for
# a transaction that must be retried: PostgreSQL for a true serialization
# conflict, MySQL for InnoDB deadlock detection (error 1213).
SERIALIZATION_FAILURE_SQLSTATE = '40001'

# MySQL error 1205, lock wait timeout. Not a deadlock -- the other transaction
# is simply slow -- but retrying is the right response either way.
MYSQL_LOCK_WAIT_TIMEOUT = 1205
MYSQL_DEADLOCK = 1213


def _sqlstate(orig):
    """Best-effort SQLSTATE for a DBAPI exception, across drivers."""
    # psycopg2 exposes it as pgcode; PyMySQL and mysqlclient expose sqlstate on
    # the exception in newer versions, and not at all in older ones.
    for attribute in ('pgcode', 'sqlstate'):
        value = getattr(orig, attribute, None)
        if value:
            return str(value)
    return None


def _mysql_errno(orig):
    """MySQL's numeric error code, which drivers put first in args."""
    args = getattr(orig, 'args', None)
    if args and isinstance(args[0], int):
        return args[0]
    return None


def is_retryable(error):
    """
    Whether a database error means "try this transaction again".

    Engine-agnostic on purpose: the original code tested
    ``isinstance(err.orig, psycopg2.errors.SerializationFailure)``, which
    silently becomes "never retry" on MySQL -- every contended vote would
    surface as a 500 instead of being retried.
    """
    orig = getattr(error, 'orig', None)
    if orig is None:
        return False
    if _sqlstate(orig) == SERIALIZATION_FAILURE_SQLSTATE:
        return True
    return _mysql_errno(orig) in (MYSQL_DEADLOCK, MYSQL_LOCK_WAIT_TIMEOUT)


def is_foreign_key_violation(error):
    """
    Whether a database error is a foreign key constraint violation.

    PostgreSQL reports SQLSTATE 23503; MySQL reports 23000 with error 1451
    (row is referenced) or 1452 (no parent row).
    """
    orig = getattr(error, 'orig', None)
    if orig is None:
        return False
    if _sqlstate(orig) == '23503':
        return True
    return _mysql_errno(orig) in (1451, 1452)


def with_retries(tx_func):
    '''
    Keep retrying a function that involves a database transaction until it
    succeeds.

    This only retries errors that mean "the engine could not serialize this,
    try again"; all other types of exceptions are re-raised.
    '''
    for attempt in range(MAX_RETRIES):
        try:
            tx_func()
        except sqlalchemy.exc.DBAPIError as err:
            # Roll back before anything else: a failed transaction leaves the
            # session unusable, and every later statement on it would fail with
            # an unhelpful InternalError instead of the real cause.
            db.session.rollback()
            if not is_retryable(err):
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            # Brief backoff so two transactions that keep colliding do not
            # retry in lockstep forever.
            time.sleep(0.05 * (2 ** attempt))
        except Exception:
            db.session.rollback()
            raise
        else:
            return
