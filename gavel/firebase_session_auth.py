# Firebase Session Authentication for HackPSU Integration
#
# Gavel does not hold Firebase credentials, so it cannot verify a session
# cookie itself. It delegates that to the HackPSU auth server, whose
# /api/sessionUser endpoint calls admin.auth().verifySessionCookie(token, true)
# -- a real signature check plus a revocation check -- and answers 200 with a
# custom token, or 401.
#
# This replaces an earlier implementation that called jwt.decode() with
# {"verify_signature": False} and trusted the resulting `production`/`staging`
# claims directly. Anyone could mint a cookie claiming production=4 and get
# admin. AUTH_SERVER_URL was already configured; nothing ever called it.

import hashlib
import os
import threading
import time

import jwt
import requests
from flask import request, session, redirect, render_template
from werkzeug.datastructures import MultiDict
from urllib.parse import urlencode

from functools import wraps

from gavel.models import (
    Annotator,
    NoActiveHackathon,
    current_hackathon_id,
    db,
)
from gavel import app

AUTH_SERVER_URL = os.environ.get(
    'AUTH_SERVER_URL', 'http://localhost:3000/api/sessionUser')
SESSION_COOKIE_NAME = '__session'

MIN_JUDGE_ROLE = 2  # Role.TEAM   -- only users at or above this can judge
MIN_ADMIN_ROLE = 4  # Role.TECH   -- only users at or above this reach admin

# Judges' browsers poll /api/status every 5 seconds, so verifying every request
# against the auth server would put a steady multiple of the judge count on it.
# A short cache keeps that to roughly one call per judge per interval while
# still picking up a revoked session quickly.
AUTH_CACHE_TTL = int(os.environ.get('AUTH_CACHE_TTL', 60))
# Matches the auth server's SESSION_DURATION_MS (5 days).
SESSION_MAX_AGE = 5 * 24 * 60 * 60
AUTH_TIMEOUT = float(os.environ.get('AUTH_TIMEOUT', 5))

_cache = {}
_cache_lock = threading.Lock()


def _cache_key(token):
    # Never key the cache (or log) on the raw token.
    return hashlib.sha256(token.encode('utf8')).hexdigest()


def _cache_get(token):
    key = _cache_key(token)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, user_info = entry
        if expires_at < time.time():
            _cache.pop(key, None)
            return None
        return user_info


def _cache_put(token, user_info):
    key = _cache_key(token)
    with _cache_lock:
        _cache[key] = (time.time() + AUTH_CACHE_TTL, user_info)
        # Bounded cleanup so a long-running worker doesn't accumulate entries
        # for every session it has ever seen.
        if len(_cache) > 2048:
            now = time.time()
            for k, (expires_at, _) in list(_cache.items()):
                if expires_at < now:
                    _cache.pop(k, None)


def _session_token():
    """The session token, from the cookie, a bearer header, or the handoff.

    The auth server issues a cookie only for *.hackpsu.org. For any other
    origin -- localhost, Vercel previews -- it cannot set a readable cookie, so
    buildReturnUrl() appends the token to the redirect as `?authToken=`
    instead. Gavel only ever read the cookie, so that half of the handshake was
    never implemented and logging in anywhere but production simply looped.
    Accept all three, exactly as the auth server offers them.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        return token
    header = request.headers.get('Authorization', '')
    if header.startswith('Bearer '):
        return header[7:]
    return request.args.get('authToken') or None


@app.before_request
def _accept_auth_token_handoff():
    """
    Complete the non-production login handoff.

    When the auth server sends the user back with `?authToken=`, verify it once
    and store it as a cookie, then redirect to the same URL without the token.
    Stripping it matters: a session token left in the URL ends up in browser
    history, bookmarks and any Referer header the page later sends.

    The token is verified before anything is stored, so this grants no access
    that presenting the same token as a cookie would not.
    """
    token = request.args.get('authToken')
    if not token:
        return None

    if verify_hackpsu_session() is None:
        # Fall through without the cookie; the route's own decorator will send
        # them back to the login page.
        return None

    stripped = MultiDict(
        (k, v) for k, v in request.args.items(multi=True) if k != 'authToken')
    target = request.path + (('?' + urlencode(list(stripped.items(multi=True))))
                             if stripped else '')

    response = redirect(target)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        samesite='Lax',
        secure=request.is_secure,
        max_age=SESSION_MAX_AGE,
    )
    return response


def _decode_unverified(token):
    """Read a JWT payload without checking its signature.

    Only ever called on tokens the auth server has already vouched for: the
    session cookie after a 200 from /api/sessionUser, and the custom token that
    response returned over TLS. The signature check is the auth server's job;
    this just reads fields out of a blob already established as authentic.
    """
    try:
        return jwt.decode(token, options={'verify_signature': False})
    except Exception:
        return None


def _privilege_from_custom_token(payload, auth_env):
    """Role level out of the custom token the auth server minted.

    createCustomToken(uid, additionalClaims) nests everything under `claims`,
    and sessionUser builds additionalClaims as
    {...userRecord.customClaims, claims: {production, staging}} -- so the
    defaulted values sit one level deeper than the spread ones.
    """
    claims = (payload or {}).get('claims') or {}
    nested = claims.get('claims') or {}
    for source in (nested, claims):
        value = source.get(auth_env)
        if isinstance(value, int):
            return value
    return 0



def _users_api_url(uid):
    """Derive GET /users/<uid> from the configured projects URL."""
    base = os.environ.get(
        'HACKPSU_API_URL',
        'https://apiv3.hackpsu.org/judging/projects').split('/judging/')[0]
    return '%s/users/%s' % (base.rstrip('/'), uid)


def _name_from_hackpsu_api(uid):
    """
    The judge's real name, from the HackPSU user record.

    Firebase only carries a `name` claim when the account has a displayName
    set, and most HackPSU organizers never set one -- so Gavel fell back to the
    email local-part and the judge list filled up with "rxr5630" and "jwb6636".
    HackPSU knows the actual name: it is on the user record from registration.

    GET /users/:id is declared @Roles(Role.NONE), which roles.guard treats as
    "no authentication required" (its catch block returns true when passport
    throws), so this needs no credential.

    Best-effort: any failure falls back to the previous behaviour.
    """
    if not uid:
        return None
    try:
        response = requests.get(_users_api_url(uid), timeout=AUTH_TIMEOUT)
        if response.status_code != 200:
            return None
        user = response.json() or {}
    except (requests.RequestException, ValueError):
        return None

    full = ' '.join(
        part for part in (user.get('firstName'), user.get('lastName')) if part
    ).strip()
    return full or None


def verify_hackpsu_session():
    """Authenticate the caller, or return None.

    Returns a user dict on success. Fails closed: any error, timeout or
    non-200 from the auth server means unauthenticated.
    """
    token = _session_token()
    if not token:
        return None

    cached = _cache_get(token)
    if cached is not None:
        return cached

    try:
        response = requests.get(
            AUTH_SERVER_URL,
            headers={'Authorization': 'Bearer %s' % token},
            cookies={SESSION_COOKIE_NAME: token},
            timeout=AUTH_TIMEOUT,
        )
    except requests.RequestException as e:
        app.logger.warning('auth server unreachable: %s', e)
        return None

    if response.status_code != 200:
        return None

    try:
        custom_token = response.json().get('customToken')
    except ValueError:
        app.logger.warning('auth server returned a non-JSON body')
        return None
    if not custom_token:
        return None

    auth_env = os.environ.get('AUTH_ENVIRONMENT', 'production')
    privilege = _privilege_from_custom_token(
        _decode_unverified(custom_token), auth_env)

    # Identity comes from the session cookie, which the auth server has now
    # confirmed is authentic. The custom token carries the uid but no email,
    # and Gavel keys judges by email.
    claims = _decode_unverified(token) or {}
    email = claims.get('email') or ''
    uid = claims.get('user_id') or claims.get('sub')
    if not uid and not email:
        return None

    # Prefer the Firebase display name, then the HackPSU user record, and only
    # then the email local-part. The middle step is what stops the judge list
    # reading as a column of PSU access IDs.
    name = claims.get('name') or claims.get('displayName')
    if not name:
        name = _name_from_hackpsu_api(uid)
    fallback_name = email.split('@')[0] if email else None

    user_info = {
        'uid': uid,
        'email': email,
        'displayName': name or fallback_name or 'Unknown User',
        # Lets the sync avoid overwriting a good stored name with a guess.
        'name_is_fallback': not name,
        'privilege': privilege,
    }
    _cache_put(token, user_info)
    return user_info


def extract_user_privilege(user_data):
    """Role level for the configured environment."""
    return user_data.get('privilege', 0)


def check_judge_permission(user_data):
    min_role = int(os.environ.get('MIN_JUDGE_ROLE', MIN_JUDGE_ROLE))
    return extract_user_privilege(user_data) >= min_role


def check_admin_permission(user_data):
    min_role = int(os.environ.get('MIN_ADMIN_ROLE', MIN_ADMIN_ROLE))
    return extract_user_privilege(user_data) >= min_role


def _login_redirect():
    """
    Send the user to the auth server, asking to be returned here.

    returnTo is built from GAVEL_URL rather than request.url. Cloud Run serves
    a service on two hostnames -- the canonical
    <service>-<project-number>.<region>.run.app and a legacy
    <service>-<hash>.a.run.app -- and the auth server's allowlist can only
    recognise the first, because the legacy form carries a random hash instead
    of the project number. A judge who arrived on the legacy hostname would
    otherwise hand the auth server a returnTo it rejects, and be bounced to
    hackpsu.org instead of back here.

    Pinning the canonical origin also means a custom domain keeps working the
    moment GAVEL_URL points at it.
    """
    auth_login_url = os.environ.get(
        'AUTH_LOGIN_URL', 'http://localhost:3000/login')

    canonical = os.environ.get('GAVEL_URL')
    if canonical:
        return_to = canonical.rstrip('/') + request.full_path.rstrip('?')
    else:
        return_to = request.url

    return redirect('%s?returnTo=%s' % (auth_login_url, return_to))


def _no_hackathon_error():
    return render_template(
        'error.html',
        message='No hackathon is currently active. An admin needs to sync one '
                'from the HackPSU API before judging can start.'
    ), 503


def hackpsu_auth_required(f):
    """Require HackPSU authentication with judge permissions"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_data = verify_hackpsu_session()
        if not user_data:
            return _login_redirect()

        if not check_judge_permission(user_data):
            privilege = extract_user_privilege(user_data)
            return render_template(
                'error.html',
                message='You need organizer permissions (level %s+) to access '
                        'judging. Your current level: %s'
                        % (os.environ.get('MIN_JUDGE_ROLE', MIN_JUDGE_ROLE),
                           privilege)
            ), 403

        try:
            annotator = sync_annotator_from_auth_server(user_data)
        except NoActiveHackathon:
            return _no_hackathon_error()

        if not annotator or not annotator.active:
            return render_template(
                'error.html',
                message='Your judging account is not active. Please contact an '
                        'admin.'
            ), 403

        session['annotator_id'] = annotator.id
        return f(*args, **kwargs)
    return decorated


def hackpsu_admin_required(f):
    """Require HackPSU authentication with admin permissions"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_data = verify_hackpsu_session()
        if not user_data:
            return _login_redirect()

        if not check_admin_permission(user_data):
            return render_template(
                'error.html',
                message='Admin access required. You need permission level %s+.'
                        % os.environ.get('MIN_ADMIN_ROLE', MIN_ADMIN_ROLE)
            ), 403

        # Admins may also judge, so give them a judge row when one is possible.
        # Unlike the judge path, a missing hackathon is not fatal here: the
        # admin pages are where you go to fix that.
        try:
            annotator = sync_annotator_from_auth_server(user_data)
            if annotator:
                session['annotator_id'] = annotator.id
        except NoActiveHackathon:
            session.pop('annotator_id', None)

        return f(*args, **kwargs)
    return decorated


def sync_annotator_from_auth_server(user_data):
    """Create or update this person's judge row for the active hackathon."""
    email = user_data.get('email')
    if not email:
        return None

    privilege = extract_user_privilege(user_data)
    # Mirrors apiv3's Role enum (common/gcp/auth/firebase-auth.types.ts).
    # 5 was missing, so finance staff showed up as "Role Level 5".
    role_names = {
        0: 'User',
        1: 'Hacker',
        2: 'Organizer',
        3: 'Executive',
        4: 'Admin',
        5: 'Finance',
    }
    role_description = role_names.get(privilege, 'Role Level %s' % privilege)

    hackathon_id = current_hackathon_id()  # raises NoActiveHackathon
    min_role = int(os.environ.get('MIN_JUDGE_ROLE', MIN_JUDGE_ROLE))

    # Scoped to the active hackathon: a judge from last year's event gets a
    # fresh row, with fresh priors and no memory of which projects they saw.
    annotator = Annotator.by_email(email, hackathon_id)

    if not annotator:
        annotator = Annotator(
            name=user_data.get('displayName') or email.split('@')[0],
            email=email,
            description=role_description,
            hackathon_id=hackathon_id,
        )
        annotator.active = privilege >= min_role
        db.session.add(annotator)
    else:
        # Only replace the stored name with something at least as good. The old
        # code assigned unconditionally, so a judge whose real name had been
        # recorded got overwritten with the email local-part on their next
        # request if the API lookup happened to fail.
        incoming = user_data.get('displayName')
        if incoming and not (user_data.get('name_is_fallback') and
                             annotator.name != incoming):
            annotator.name = incoming
        annotator.description = role_description
        annotator.active = privilege >= min_role
    db.session.commit()

    return annotator


@app.context_processor
def inject_user_data():
    """Make user data available to all templates for PostHog identification"""
    user_data = verify_hackpsu_session()
    if user_data:
        return {
            'user_uid': user_data.get('uid'),
            'user_email': user_data.get('email'),
        }
    return {'user_uid': None, 'user_email': None}
