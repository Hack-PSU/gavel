from gavel import app
from humanize import naturaltime
import calendar
import datetime as dt

# Every timestamp in the database is naive UTC: the models default to
# datetime.utcnow(), and comparisons inside the app are UTC-against-UTC, so
# they stay consistent. The bug was at the display boundary, where a naive UTC
# value met a formatter that assumed local time.


@app.template_filter('utcdatetime_local')
def _jinja2_filter_datetime_local(value):
    '''
    Render a stored UTC timestamp as "5 minutes ago".

    humanize compares against datetime.now() -- local time -- unless told
    otherwise. Handing it a naive UTC value from the database made every recent
    timestamp read as being in the future by exactly the UTC offset: a vote
    cast seconds ago showed as "4 hours from now" in US Eastern. Anchoring the
    comparison to utcnow() puts both sides in the same frame.
    '''
    if value is None:
        return 'None'
    return naturaltime(value, when=dt.datetime.utcnow())


@app.template_filter('utcdatetime_epoch')
def _jinja2_filter_datetime_epoch(value):
    '''
    Epoch seconds for a stored UTC timestamp, used as a table sort key.

    strftime('%s') is a platform extension that interprets a naive datetime as
    local time, so it was skewed by the same UTC offset -- and it is undefined
    on platforms whose strftime lacks %s. timegm() reads the value as UTC,
    which is what it actually is.
    '''
    if value is None:
        return 0
    return calendar.timegm(value.utctimetuple())
