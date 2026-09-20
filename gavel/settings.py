import gavel.constants as constants
import os
import yaml

BASE_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(BASE_DIR, '..', 'config.yaml')

NONE_SENTINEL = object()  # because default may be None

class Config(object):

    def __init__(self, config_file):
        if not _bool(os.environ.get('IGNORE_CONFIG_FILE', False)):
            with open(config_file) as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = {}

    # checks for an environment variable first, then an entry in the config file,
    # and then falls back to default
    def get(self, name, env_names=None, default=NONE_SENTINEL):
        setting = None
        if env_names is not None:
            if not isinstance(env_names, list):
                env_names = [env_names]
            for env_name in env_names:
                setting = os.environ.get(env_name, None)
                if setting is not None:
                    break
        if setting is None:
            setting = self._config.get(name, None)
        if setting is None:
            if default is not NONE_SENTINEL:
                return default
            else:
                raise LookupError('Cannot find value for setting %s' % name)
        return setting

def _bool(truth_value):
    if isinstance(truth_value, bool):
        return truth_value
    if isinstance(truth_value, int):
        return bool(truth_value)
    if isinstance(truth_value, str):
        if truth_value.isnumeric():
            return bool(int(truth_value))
        lower = truth_value.lower()
        return lower.startswith('t') or lower.startswith('y') # accepts things like 'yes', 'True', ...
    raise ValueError('invalid type for bool coercion')

def _list(item):
    if isinstance(item, list):
        return item
    return [item]

# SQLAlchemy used to support URIs starting with 'postgres://', but it now
# requires that they start with 'postgresql://'. Heroku sets the environment
# variable 'DATABASE_URL' to 'postgres://...', so this breaks our app. This
# function fixes the DB URI.
def _postgres_uri(uri):
    if uri.startswith('postgres://'):
        return uri.replace('postgres://', 'postgresql://', 1)
    return uri


def _mysql_uri_from_parts():
    """
    Build a database URI from the MYSQL_* variables, the way apiv3 does.

    Cloud Run injects each of these from its own Secret Manager entry, so the
    database password lives in exactly one secret that both services read --
    rather than being duplicated into a second secret holding a whole URL,
    where it would then have to be rotated in two places.

    Returns None when the parts are not all present, so DATABASE_URL and
    DB_URI keep working unchanged everywhere else.
    """
    user = os.environ.get('MYSQL_USER')
    password = os.environ.get('MYSQL_PASSWORD')
    database = os.environ.get('MYSQL_DATABASE')
    if not (user and password and database):
        return None

    from urllib.parse import quote_plus
    credentials = '%s:%s' % (quote_plus(user), quote_plus(password))

    socket_path = os.environ.get('MYSQL_SOCKET_PATH')
    if socket_path:
        # Cloud Run mounts the Cloud SQL socket rather than exposing a host,
        # so there is no netloc -- the path goes in a query parameter.
        return 'mysql+pymysql://%s@/%s?unix_socket=%s' % (
            credentials, database, quote_plus(socket_path))

    host = os.environ.get('MYSQL_HOST', '127.0.0.1')
    port = os.environ.get('MYSQL_PORT', '3306')
    return 'mysql+pymysql://%s@%s:%s/%s' % (credentials, host, port, database)


c = Config(CONFIG_FILE)

# note: this should be kept in sync with 'config.template.yaml' and
# 'config.vagrant.yaml'
SERVER_NAME =          c.get('server_name',     'SERVER_NAME',               default=None)
PROXY =          _bool(c.get('proxy',           'PROXY',                     default=False))
# Optional: the HTTP-basic admin login this guarded was replaced by HackPSU
# Firebase auth, and nothing applies utils.requires_auth any more. Leaving it
# mandatory meant every deployment had to supply a secret that protected
# nothing -- and the app refused to start without one.
ADMIN_PASSWORD =       c.get('admin_password',  'ADMIN_PASSWORD',            default=None)
DB_URI = _postgres_uri(c.get('db_uri',          ['DATABASE_URL', 'DB_URI'],
                             default=_mysql_uri_from_parts()
                                     or 'postgresql://localhost/gavel'))
BROKER_URI =           c.get('broker_uri',      ['REDIS_URL', 'BROKER_URI'], default='redis://localhost:6379/0')
SECRET_KEY =           c.get('secret_key',      'SECRET_KEY')
MIN_VIEWS =        int(c.get('min_views',       'MIN_VIEWS',                 default=2))
TIMEOUT =        float(c.get('timeout',         'TIMEOUT',                   default=5.0)) # in minutes
WELCOME_MESSAGE =      c.get('welcome_message',                              default=constants.DEFAULT_WELCOME_MESSAGE)
CLOSED_MESSAGE =       c.get('closed_message',                               default=constants.DEFAULT_CLOSED_MESSAGE)
DISABLED_MESSAGE =     c.get('disabled_message',                             default=constants.DEFAULT_DISABLED_MESSAGE)
LOGGED_OUT_MESSAGE =   c.get('logged_out_message',                           default=constants.DEFAULT_LOGGED_OUT_MESSAGE)
WAIT_MESSAGE =         c.get('wait_message',                                 default=constants.DEFAULT_WAIT_MESSAGE)
DISABLE_EMAIL =  _bool(c.get('disable_email',   'DISABLE_EMAIL',             default=False))
EMAIL_HOST =           c.get('email_host',      'EMAIL_HOST',                default='smtp.gmail.com')
EMAIL_PORT =       int(c.get('email_port',      'EMAIL_PORT',                default=587))
# Optional: email is deprecated. Judge invites went out through Celery, which
# needs a broker and a worker process that no longer exist. Leaving these
# mandatory meant every deployment had to invent placeholder values -- the
# production image literally set EMAIL_USER=unused -- or the app would not
# start. The email paths check DISABLE_EMAIL before using them.
EMAIL_FROM =           c.get('email_from',      'EMAIL_FROM',                default=None)
EMAIL_USER =           c.get('email_user',      'EMAIL_USER',                default=None)
EMAIL_PASSWORD =       c.get('email_password',  'EMAIL_PASSWORD',            default=None)
EMAIL_AUTH_MODE =      c.get('email_auth_mode', 'EMAIL_AUTH_MODE',           default='tls').lower()
EMAIL_CC =       _list(c.get('email_cc',        'EMAIL_CC',                  default=[]))
EMAIL_SUBJECT =        c.get('email_subject',                                default=constants.DEFAULT_EMAIL_SUBJECT)
EMAIL_BODY =           c.get('email_body',                                   default=constants.DEFAULT_EMAIL_BODY)
SEND_STATS =     _bool(c.get('send_stats',      'SEND_STATS',                default=True))
USE_SENDGRID =   _bool(c.get('use_sendgrid',    'USE_SENDGRID',              default=False))
SENDGRID_API_KEY =     c.get('sendgrid_api_key', 'SENDGRID_API_KEY',         default=None)
TIMER_DURATION =   int(c.get('timer_duration',   'TIMER_DURATION',            default=3)) # in minutes

# Database pool, per gunicorn worker. The total a deployment can open is
# (instances x workers x (DB_POOL_SIZE + DB_MAX_OVERFLOW)); keep that well
# under the server's max_connections, which is shared with other services.
DB_POOL_SIZE =     int(c.get('db_pool_size',    'DB_POOL_SIZE',              default=5))
DB_MAX_OVERFLOW =  int(c.get('db_max_overflow', 'DB_MAX_OVERFLOW',           default=2))
DB_POOL_RECYCLE =  int(c.get('db_pool_recycle', 'DB_POOL_RECYCLE',           default=1800)) # seconds
