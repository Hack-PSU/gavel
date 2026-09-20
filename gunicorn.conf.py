"""
Gunicorn configuration for Cloud Run.

Cloud Run gives the container one request-serving process, injects the port to
listen on, and terminates TLS in front. Everything here follows from that plus
the size of the shared Cloud SQL instance.
"""

import os

# Cloud Run injects PORT and routes to it. Hardcoding 5000 -- as the old image
# did -- means the container never receives traffic.
bind = '0.0.0.0:%s' % os.environ.get('PORT', '8080')

# Threads, not more processes.
#
# The old `-w 16` was sized as if the container owned the machine. Each worker
# keeps its own SQLAlchemy pool, so sixteen of them could hold 240 connections
# from a single instance against a 280-connection server shared with apiv3 --
# and Cloud Run runs several instances. Two threaded workers serve the same
# concurrency on one vCPU while bounding the pool at
# workers x (DB_POOL_SIZE + DB_MAX_OVERFLOW) = 14 per instance.
workers = int(os.environ.get('WEB_CONCURRENCY', '2'))
threads = int(os.environ.get('WEB_THREADS', '8'))
worker_class = 'gthread'

# Gavel's work is database round trips, not CPU, so threads block on I/O and
# release the GIL. gevent would also fit but needs monkey-patching before the
# drivers are imported, which is easy to get subtly wrong.

# Cloud Run's own request timeout is 300s by default; staying under it means
# gunicorn is the one to give up, with a log line, rather than the platform.
timeout = int(os.environ.get('WEB_TIMEOUT', '120'))
graceful_timeout = 30

# Cloud Run reuses connections, so a short keepalive would churn them.
keepalive = 65

# Load the app before forking: the import builds the Jinja environment and the
# asset bundle once, rather than once per worker, which shortens cold starts.
preload_app = True

accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info')
# Cloud Run's load balancer is the immediate peer, so log the forwarded client.
access_log_format = '%({x-forwarded-for}i)s "%(r)s" %(s)s %(b)s %(D)sus'
