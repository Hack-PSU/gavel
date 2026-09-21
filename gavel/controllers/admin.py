from gavel import app
from gavel.models import *
from gavel.constants import *
import gavel.settings as settings
import gavel.utils as utils
import gavel.stats as stats
import gavel.analytics as analytics
from gavel.firebase_session_auth import hackpsu_admin_required
from gavel.project_sync import (
    sync_projects_from_api,
    sync_active_hackathon,
    maybe_sync_projects,
)
from flask import (
    redirect,
    render_template,
    request,
    url_for,
    flash,
)
import urllib.parse
import xlrd

ALLOWED_EXTENSIONS = set(['csv', 'xlsx', 'xls'])

@app.route('/admin/')
@hackpsu_admin_required
def admin():
    stats.check_send_telemetry()
    maybe_sync_projects()

    # Degrade rather than 503: this page is where an admin starts a hackathon,
    # so it has to render when there isn't one.
    if current_hackathon_id(required=False) is None:
        return render_template(
            'admin_setup.html',
            hackathons=Hackathon.query.order_by(Hackathon.name).all(),
        )

    hackathon_id = current_hackathon_id()
    annotators = Annotator.query_current().order_by(Annotator.id).all()
    items = Item.query_current().order_by(Item.id).all()

    # Load every comparison once, as four columns rather than full ORM
    # objects, and hand it to each analytic below. Each of them used to run its
    # own Decision.query.all(), so one page load scanned the decision table
    # five times and built five sets of objects it immediately discarded.
    comparisons = analytics.load_comparisons(hackathon_id)

    counts = {}
    item_counts = {}
    for winner_id, loser_id, annotator_id, _time in comparisons:
        counts[annotator_id] = counts.get(annotator_id, 0) + 1
        item_counts[winner_id] = item_counts.get(winner_id, 0) + 1
        item_counts[loser_id] = item_counts.get(loser_id, 0) + 1

    view_counts = view_counts_by_item(hackathon_id)
    skipped = skip_counts_by_item(hackathon_id)

    # settings
    setting_closed = Setting.value_of(SETTING_CLOSED) == SETTING_TRUE

    # Graph visualization data
    G = analytics.build_comparison_graph(items, comparisons)
    graph_data = analytics.generate_graph_data_for_visualization(G)

    # New analytics data
    coverage_matrix = analytics.get_coverage_matrix(items, comparisons)
    voting_timeline = analytics.get_voting_timeline(hours=2, comparisons=comparisons)
    statistical_summary = analytics.get_statistical_summary(
        items, annotators, comparisons)

    return render_template(
        'admin.html',
        # by_id, not current(): resolving the id above already loaded this row
        # into the session, so this is an identity-map hit rather than a query.
        hackathon=Hackathon.by_id(hackathon_id),
        annotators=annotators,
        counts=counts,
        item_counts=item_counts,
        view_counts=view_counts,
        skipped=skipped,
        items=items,
        votes=len(comparisons),
        setting_closed=setting_closed,
        graph_data=graph_data,
        coverage_matrix=coverage_matrix,
        voting_timeline=voting_timeline,
        statistical_summary=statistical_summary,
    )


def view_counts_by_item(hackathon_id):
    """
    How many judges have seen each project, as one GROUP BY.

    The template renders `item.viewed | length` per row, and the skip
    calculation walked `item.viewed` for every project -- each a lazy load, so
    300 projects meant 300 round trips before the page could render.
    """
    rows = db.session.query(
        view_table.c.item_id, db.func.count()
    ).join(
        Item, Item.id == view_table.c.item_id
    ).filter(
        Item.hackathon_id == hackathon_id
    ).group_by(view_table.c.item_id).all()
    return dict(rows)


def skip_counts_by_item(hackathon_id):
    """
    How many judges skipped each project: ignored it without ever viewing it.

    This was a nested Python loop over every judge's `ignore` collection --
    another lazy load per judge -- cross-referenced against a dict of every
    project's viewers. The same question is one anti-join.
    """
    view_alias = view_table.alias('v')
    rows = db.session.query(
        ignore_table.c.item_id, db.func.count()
    ).join(
        Item, Item.id == ignore_table.c.item_id
    ).outerjoin(
        view_alias,
        (view_alias.c.item_id == ignore_table.c.item_id) &
        (view_alias.c.annotator_id == ignore_table.c.annotator_id)
    ).filter(
        (Item.hackathon_id == hackathon_id) &
        (view_alias.c.item_id.is_(None))
    ).group_by(ignore_table.c.item_id).all()
    return dict(rows)


@app.route('/admin/item', methods=['POST'])
@hackpsu_admin_required
def item():
    action = request.form['action']
    if action == 'Submit':
        data = parse_upload_form()
        if data:
            # validate data
            for index, row in enumerate(data):
                if len(row) != 3:
                    return utils.user_error('Bad data: row %d has %d elements (expecting 3)' % (index + 1, len(row)))
            def tx():
                for row in data:
                    _item = Item(*row)
                    db.session.add(_item)
                db.session.commit()
            with_retries(tx)
    elif action == 'Prioritize' or action == 'Cancel':
        item_id = request.form['item_id']
        target_state = action == 'Prioritize'
        def tx():
            Item.by_id(item_id).prioritized = target_state
            db.session.commit()
        with_retries(tx)
    elif action == 'Disable' or action == 'Enable':
        item_id = request.form['item_id']
        target_state = action == 'Enable'
        def tx():
            Item.by_id(item_id).active = target_state
            db.session.commit()
        with_retries(tx)
    elif action == 'Delete':
        item_id = request.form['item_id']
        try:
            def tx():
                db.session.execute(ignore_table.delete(ignore_table.c.item_id == item_id))
                Item.query_current().filter_by(id=item_id).delete()
                db.session.commit()
            with_retries(tx)
        except IntegrityError as e:
            if is_foreign_key_violation(e):
                return utils.server_error("Projects can't be deleted once they have been voted on by a judge. You can use the 'disable' functionality instead, which has a similar effect, preventing the project from being shown to judges.")
            else:
                return utils.server_error(str(e))
    return redirect(url_for('admin'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_upload_form():
    f = request.files.get('file')
    data = []
    if f and allowed_file(f.filename):
        extension = str(f.filename.rsplit('.', 1)[1].lower())
        if extension == "xlsx" or extension == "xls":
            workbook = xlrd.open_workbook(file_contents=f.read())
            worksheet = workbook.sheet_by_index(0)
            data = list(utils.cast_row(worksheet.row_values(rx, 0, 3)) for rx in range(worksheet.nrows) if worksheet.row_len(rx) == 3)
        elif extension == "csv":
            data = utils.data_from_csv_string(f.read().decode("utf-8"))
    else:
        csv = request.form['data']
        data = utils.data_from_csv_string(csv)
    return data


@app.route('/admin/item_patch', methods=['POST'])
@hackpsu_admin_required
def item_patch():
    item_id = request.form['item_id']
    # Checked out here rather than inside tx(): a `return` from the retried
    # transaction function is discarded, so the not-found response never
    # reached the admin.
    if not Item.by_id(item_id):
        return utils.user_error('Item %s not found ' % item_id)

    def tx():
        _item = Item.by_id(item_id)
        if _item is None:
            return
        if 'location' in request.form:
            _item.location = request.form['location']
        if 'name' in request.form:
            _item.name = request.form['name']
        if 'description' in request.form:
            _item.description = request.form['description']
        db.session.commit()
    with_retries(tx)
    # `item` at module scope is the /admin/item route function, not a row --
    # `item.id` raised AttributeError and 500'd every project edit.
    return redirect(url_for('item_detail', item_id=item_id))

@app.route('/admin/annotator', methods=['POST'])
@hackpsu_admin_required
def annotator():
    action = request.form['action']
    if action == 'Submit':
        data = parse_upload_form()
        added = []
        if data:
            # validate data
            for index, row in enumerate(data):
                if len(row) != 3:
                    return utils.user_error('Bad data: row %d has %d elements (expecting 3)' % (index + 1, len(row)))
            def tx():
                for row in data:
                    annotator = Annotator(*row)
                    added.append(annotator)
                    db.session.add(annotator)
                db.session.commit()
            with_retries(tx)
            try:
                email_invite_links(added)
            except Exception as e:
                return utils.server_error(str(e))
    elif action == 'Email':
        annotator_id = request.form['annotator_id']
        try:
            email_invite_links(Annotator.by_id(annotator_id))
        except Exception as e:
            return utils.server_error(str(e))
    elif action == 'Disable' or action == 'Enable':
        annotator_id = request.form['annotator_id']
        target_state = action == 'Enable'
        def tx():
            Annotator.by_id(annotator_id).active = target_state
            db.session.commit()
        with_retries(tx)
    elif action == 'Delete':
        annotator_id = request.form['annotator_id']
        try:
            def tx():
                db.session.execute(ignore_table.delete(ignore_table.c.annotator_id == annotator_id))
                Annotator.query_current().filter_by(id=annotator_id).delete()
                db.session.commit()
            with_retries(tx)
        except IntegrityError as e:
            if is_foreign_key_violation(e):
                return utils.server_error("Judges can't be deleted once they have voted on a project. You can use the 'disable' functionality instead, which has a similar effect, locking out the judge and preventing them from voting on any other projects.")
            else:
                return utils.server_error(str(e))
    return redirect(url_for('admin'))

@app.route('/admin/setting', methods=['POST'])
@hackpsu_admin_required
def setting():
    key = request.form['key']
    if key == 'closed':
        action = request.form['action']
        new_value = SETTING_TRUE if action == 'Close' else SETTING_FALSE
        Setting.set(SETTING_CLOSED, new_value)
        db.session.commit()
    return redirect(url_for('admin'))

@app.route('/admin/hackathon', methods=['POST'])
@hackpsu_admin_required
def hackathon():
    """
    Set the active hackathon: the tenant that owns all judging data.

    'Sync' adopts whichever hackathon the HackPSU API reports as active.
    'Activate' switches to one Gavel already knows about, which is how you
    reopen a past event without touching the API.
    """
    action = request.form['action']
    if action == 'Sync':
        try:
            hackathon_id = sync_active_hackathon()
        except Exception as e:
            return utils.server_error('Failed to sync hackathon: %s' % e)
        if not hackathon_id:
            return utils.server_error(
                'The HackPSU API did not report an active hackathon. Check '
                'HACKPSU_API_KEY and HACKPSU_API_URL.')
    elif action == 'Activate':
        target = request.form['hackathon_id']
        existing = Hackathon.by_id(target)
        if not existing:
            return utils.user_error('Hackathon %s not found' % target)
        def tx():
            Hackathon.activate(existing.id, existing.name)
            db.session.commit()
        with_retries(tx)
        forget_current_hackathon()
    return redirect(url_for('admin'))

@app.route('/admin/sync-projects', methods=['POST'])
@hackpsu_admin_required
def sync_projects():
    """Manually trigger project sync from HackPSU API"""
    try:
        sync_projects_from_api()
        return redirect(url_for('admin'))
    except Exception as e:
        return utils.server_error(f'Failed to sync projects: {str(e)}')

@app.route('/admin/item/<item_id>/')
@hackpsu_admin_required
def item_detail(item_id):
    item = Item.by_id(item_id)
    if not item:
        return utils.user_error('Item %s not found ' % item_id)
    else:
        assigned = Annotator.query_current().filter(Annotator.next == item).all()
        viewed_ids = {i.id for i in item.viewed}
        if viewed_ids:
            skipped = Annotator.query_current().filter(
                Annotator.ignore.contains(item) & ~Annotator.id.in_(viewed_ids)
            )
        else:
            skipped = Annotator.query_current().filter(Annotator.ignore.contains(item))

        # Get skip reasons (and the judge's explanation, where there is one)
        skip_records = Skip.query.filter_by(item_id=item_id).all()
        skip_reasons = {s.annotator_id: s.reason for s in skip_records}
        skip_notes = {s.annotator_id: s.note for s in skip_records if s.note}

        return render_template(
            'admin_item.html',
            item=item,
            assigned=assigned,
            skipped=skipped,
            skip_reasons=skip_reasons,
            skip_notes=skip_notes
        )

@app.route('/admin/annotator/<annotator_id>/')
@hackpsu_admin_required
def annotator_detail(annotator_id):
    annotator = Annotator.by_id(annotator_id)
    if not annotator:
        return utils.user_error('Annotator %s not found ' % annotator_id)
    else:
        seen = Item.query_current().filter(Item.viewed.contains(annotator)).all()
        ignored_ids = {i.id for i in annotator.ignore}
        if ignored_ids:
            skipped = Item.query_current().filter(
                Item.id.in_(ignored_ids) & ~Item.viewed.contains(annotator)
            )
        else:
            skipped = []

        # Get skip reasons (and the judge's explanation, where there is one)
        skip_records = Skip.query.filter_by(annotator_id=annotator_id).all()
        skip_reasons = {s.item_id: s.reason for s in skip_records}
        skip_notes = {s.item_id: s.note for s in skip_records if s.note}

        return render_template(
            'admin_annotator.html',
            annotator=annotator,
            login_link=annotator_link(annotator),
            seen=seen,
            skipped=skipped,
            skip_reasons=skip_reasons,
            skip_notes=skip_notes
        )

def annotator_link(annotator):
        import os
        gavel_url = os.environ.get('GAVEL_URL', 'http://localhost:5000')
        return gavel_url

def email_invite_links(annotators):
    if settings.DISABLE_EMAIL or annotators is None:
        return
    if not settings.EMAIL_FROM:
        # Deprecated path, left in place but no longer configured by default.
        # Say so rather than failing deep inside the mail transport.
        raise RuntimeError(
            'Email is not configured (EMAIL_FROM unset). Judge invites are '
            'deprecated; judges sign in through HackPSU auth instead.')
    if not isinstance(annotators, list):
        annotators = [annotators]

    emails = []
    for annotator in annotators:
        link = annotator_link(annotator)
        raw_body = settings.EMAIL_BODY.format(name=annotator.name, link=link)
        body = '\n\n'.join(utils.get_paragraphs(raw_body))
        emails.append((annotator.email, settings.EMAIL_SUBJECT, body))

    if settings.USE_SENDGRID and settings.SENDGRID_API_KEY != None:
        utils.send_sendgrid_emails(emails)
    else:
        utils.send_emails.delay(emails)
