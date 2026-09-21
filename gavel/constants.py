ANNOTATOR_ID = 'annotator_id'
TELEMETRY_URL = 'https://telemetry.anish.io/api/v1/submit'
TELEMETRY_DELTA = 20 * 60 # seconds
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"

# Judge-supplied free text is stored in unbounded TEXT columns, so there is no
# length limit to enforce anywhere -- see migration 0001. The original bug was a
# bounded VARCHAR, and the fix is an unbounded column, not a smaller cap.

# Skip reasons offered in the UI. Validated server-side so the stored value is
# always one of these; SKIP_REASONS_REQUIRING_NOTE additionally require the
# judge to say what they mean.
SKIP_REASONS = {
    'cannot_find': 'Cannot find project',
    'conflict_of_interest': 'Conflict of interest',
    'insufficient_info': 'Insufficient information',
    'technical_issues': 'Technical issues',
    'other': 'Other',
}
SKIP_REASONS_REQUIRING_NOTE = {'other'}

# Setting
# keys
SETTING_CLOSED = 'closed' # boolean
SETTING_TELEMETRY_LAST_SENT = 'telemetry_sent_time' # integer
SETTING_LAST_PROJECT_SYNC = 'last_project_sync' # unix timestamp, per hackathon
SETTING_DEMO_RESTORE_TO = 'demo_restore_to' # hackathon to reactivate after a demo
SETTING_DEMO_EXPIRES_AT = 'demo_expires_at' # unix timestamp; demo self-terminates
# values
SETTING_TRUE = 'true'
SETTING_FALSE = 'false'

# Defaults
# these can be overridden via the config file
DEFAULT_WELCOME_MESSAGE = '''
Welcome to Gavel.

**Please read this important message carefully before continuing.**

Gavel is a fully automated expo judging system that both tells you where to go
and collects your votes.

The system is based on the model of pairwise comparison. You'll start off by
looking at a single submission, and then for every submission after that,
you'll decide whether it's better or worse than the one you looked at
**immediately beforehand**.

If at any point, you can't find a particular submission, you can click the
'Skip' button and you will be assigned a new project. **Please don't skip
unless absolutely necessary.**

Gavel makes it really simple for you to submit votes, but please think hard
before you vote. **Once you make a decision, you can't take it back**.
'''.strip()

DEFAULT_EMAIL_SUBJECT = 'Welcome to Gavel!'

DEFAULT_EMAIL_BODY = '''
Hi {name},

Welcome to Gavel, the online expo judging system. This email contains your
magic link to the judging system.

DO NOT SHARE this email with others, as it contains your personal magic link.

To access the system, visit {link}.

Once you're in, please take the time to read the welcome message and
instructions before continuing.
'''.strip()

DEFAULT_CLOSED_MESSAGE = '''
The judging system is currently closed. Reload the page to try again.
'''.strip()

DEFAULT_DISABLED_MESSAGE = '''
Your account is currently disabled. Reload the page to try again.
'''.strip()

DEFAULT_LOGGED_OUT_MESSAGE = '''
You are currently logged out. Open your magic link to get started.
'''.strip()

DEFAULT_WAIT_MESSAGE = '''
Wait for a little bit and reload the page to try again.

If you've looked at all the projects already, then you're done.
'''.strip()
