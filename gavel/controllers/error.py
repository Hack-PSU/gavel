from gavel import app
from gavel.models import NoActiveHackathon
from flask import render_template

@app.errorhandler(404)
def error_404(e):
    return (
        render_template('error.html', message='Page not found.'),
        404
    )

@app.errorhandler(403)
def error_403(e):
    return (
        render_template('error.html', message='Forbidden. Go back, refresh the page, and try again.'),
        403
    )

@app.errorhandler(500)
def error_500(e):
    return (
        render_template('error.html', message='Internal server error. Go back and try again.'),
        500
    )

@app.errorhandler(NoActiveHackathon)
def error_no_active_hackathon(e):
    """
    Every judging query is scoped to an event, so with none active there is
    nothing coherent to show. Say so plainly instead of rendering an empty
    system that looks like all the data vanished.
    """
    return (
        render_template(
            'error.html',
            message='No hackathon is currently active. An admin can start one '
                    'from the admin page, which syncs the active hackathon '
                    'from the HackPSU API.'
        ),
        503
    )
