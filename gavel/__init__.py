# Copyright (c) Anish Athalye (me@anishathalye.com)
#
# This software is released under AGPLv3. See the included LICENSE.txt for
# details.

from flask import Flask
app = Flask(__name__)

import os

import gavel.settings as settings
app.config['SQLALCHEMY_DATABASE_URI'] = settings.DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = settings.SECRET_KEY
app.config['SERVER_NAME'] = settings.SERVER_NAME

# Connection pooling, sized against the database rather than the app.
#
# The shared Cloud SQL instance allows 280 connections in total and already
# serves apiv3. SQLAlchemy's defaults (pool_size 5, max_overflow 10) give each
# gunicorn worker up to 15, so the old `-w 16` would have reached 240 from a
# single container -- exhausting the instance for everything else the moment
# Cloud Run scaled past one. Keeping the per-worker ceiling small bounds the
# total at (instances x workers x POOL_MAX).
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': settings.DB_POOL_SIZE,
    'max_overflow': settings.DB_MAX_OVERFLOW,
    # Cloud SQL drops idle connections, and a pooled connection that died
    # while idle surfaces as a failed request. Check it before handing it out.
    'pool_pre_ping': True,
    # Recycle well inside the server's wait_timeout so we close connections
    # before it does.
    'pool_recycle': settings.DB_POOL_RECYCLE,
}

if settings.PROXY:
    from werkzeug.middleware.proxy_fix import ProxyFix
    # Cloud Run terminates TLS and appends exactly one hop to each
    # X-Forwarded-* header. Without the counts, ProxyFix trusts the whole
    # chain, so a client-supplied X-Forwarded-For would be believed.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure CORS for HackPSU auth integration
from flask_cors import CORS
allowed_origins = [
    'https://auth.hackpsu.org',
    'https://hackpsu.org',
    'http://localhost:3000',
    'http://localhost:5000',
    os.environ.get('AUTH_SERVER_URL', '').replace('/api/sessionUser', '') if os.environ.get('AUTH_SERVER_URL') else None
]
allowed_origins = [origin for origin in allowed_origins if origin]  # Filter out None values

CORS(app,
     origins=allowed_origins,
     supports_credentials=True)

from flask_assets import Environment, Bundle
assets = Environment(app)
assets.config['pyscss_style'] = 'expanded'
# pyScss is pinned at 1.3.7, which imports collections.Iterable and so cannot
# run on Python 3.10+; 1.4.0 runs but fails to parse this stylesheet's mix()
# calls. Production is on Python 3.9 and unaffected, so it stays the default --
# but a local checkout on a modern Python needs libsass, which compiles the
# same source correctly.
SCSS_FILTER = os.environ.get('SCSS_FILTER', 'pyscss')
assets.config['libsass_style'] = 'expanded'
scss = Bundle(
    'css/style.scss',
    depends='**/*.scss',
    filters=(SCSS_FILTER,),
    output='all.css'
)
assets.register('scss_all', scss)

from celery import Celery
app.config['CELERY_BROKER_URL'] = settings.BROKER_URI
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

from gavel.models import db
db.app = app
db.init_app(app)

import gavel.template_filters # registers template filters

import gavel.controllers # registers controllers

# NOTE: schema migrations are deliberately NOT run on startup. db.create_all()
# only ever CREATEs missing tables, so an existing database drifts from the
# models -- but bringing it back in line is an explicit, operator-run step:
#
#     python migrate.py status
#     python migrate.py upgrade
#
# See migrate.py at the repository root.

# Set up automatic project sync from HackPSU API
if os.environ.get('ENABLE_PROJECT_SYNC', 'true').lower() == 'true':
    from gavel.project_sync import setup_project_sync
    setup_project_sync()
