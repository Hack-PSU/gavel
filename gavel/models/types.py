"""
Column types that mean the same thing on every supported engine.

Gavel runs on PostgreSQL today and on MySQL (Cloud SQL) after the move, and the
two disagree about text in ways that matter here:

* PostgreSQL ``TEXT`` is unbounded. MySQL ``TEXT`` caps at 65,535 *bytes* --
  roughly 16k characters of emoji-bearing utf8mb4 -- and silently truncates in
  non-strict mode. Judge notes are supposed to have no length limit, which is
  the whole point of the notes fix, so on MySQL they need ``LONGTEXT``.

* MySQL cannot index a ``TEXT`` column without a prefix length, so anything
  that participates in a primary key, a foreign key or a unique index has to be
  a bounded ``VARCHAR``. Those bounds are chosen from the real shape of the
  data, not picked arbitrarily.
"""

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME
from sqlalchemy.dialects.mysql import DOUBLE as MYSQL_DOUBLE
from sqlalchemy.dialects.mysql import LONGTEXT


# Free text with no length limit on either engine.
UnboundedText = Text().with_variant(LONGTEXT(), 'mysql')

# Double precision on both engines.
#
# SQLAlchemy's Float maps to DOUBLE PRECISION on PostgreSQL but to MySQL's
# FLOAT, which is single precision -- about seven significant digits. The
# crowd-BT scores are doubles, and rounding mu from 0.0799322734941707 to
# 0.0799323 is enough to reorder projects that are close together, so a
# migration on the default type would quietly change the results.
DoubleFloat = Float().with_variant(MYSQL_DOUBLE(asdecimal=False), 'mysql')

# Timestamps that keep their microseconds.
#
# MySQL DATETIME stores whole seconds unless a fractional precision is given,
# and it *rounds* rather than truncating: 18:39:45.720777 becomes 18:39:46. The
# vote ordering within a second is real information, so ask for the six digits
# PostgreSQL already keeps.
PreciseDateTime = DateTime().with_variant(MYSQL_DATETIME(fsp=6), 'mysql')

# HackPSU hackathon ids are 32-character tokens; 64 leaves room without making
# the column too wide to index alongside another.
HACKATHON_ID_LENGTH = 64

# RFC 5321 caps an address at 254 characters.
EMAIL_LENGTH = 254

# Setting keys are short internal identifiers ('closed', 'telemetry_sent_time').
SETTING_KEY_LENGTH = 128


def hackathon_id_column():
    return String(HACKATHON_ID_LENGTH)


def email_column():
    return String(EMAIL_LENGTH)


def setting_key_column():
    return String(SETTING_KEY_LENGTH)
