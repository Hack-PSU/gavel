"""
Advanced analytics for CrowdBT algorithm performance and network analysis.
"""
from gavel.models import Item, Decision, Annotator, db, current_hackathon_id
import networkx as nx
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func


def load_comparisons(hackathon_id=None):
    """
    Every comparison in the active hackathon, as lightweight rows.

    The dashboard needs only the ids and the timestamp, so this selects four
    columns instead of hydrating full Decision objects -- and it is loaded once
    and passed to each analytic below. Previously every function on the admin
    page ran its own `Decision.query.all()`, so a single page load scanned the
    whole decision table five times over and built five sets of ORM objects
    that were thrown away immediately.

    Analytics that span events are meaningless -- crowd-BT scores, coverage and
    convergence are all relative to one set of projects and judges -- so this
    is always scoped to one hackathon.
    """
    return db.session.query(
        Decision.winner_id,
        Decision.loser_id,
        Decision.annotator_id,
        Decision.time,
    ).filter(
        Decision.hackathon_id == (hackathon_id or current_hackathon_id())
    ).order_by(Decision.time).all()


def _active(items):
    return [i for i in items if i.active]


def _resolve(items, comparisons):
    """Fall back to querying when a caller hasn't loaded the data already."""
    if items is None:
        items = Item.query_current().order_by(Item.id).all()
    if comparisons is None:
        comparisons = load_comparisons()
    return items, comparisons


def build_comparison_graph(items=None, comparisons=None):
    """
    Build a directed graph where nodes are projects and edges are comparisons.
    Edge weight represents number of times A was preferred over B.
    """
    items, comparisons = _resolve(items, comparisons)
    G = nx.DiGraph()

    # Add all active projects as nodes
    for item in _active(items):
        G.add_node(item.id, name=item.name, mu=float(item.mu), sigma_sq=float(item.sigma_sq))

    # Add edges from comparisons. Reading winner_id/loser_id off the row rather
    # than dec.winner.id avoids a lazy load per decision for any project not
    # already in the session's identity map.
    edge_weights = defaultdict(int)
    for winner_id, loser_id, _annotator_id, _time in comparisons:
        # Only add edges if both winner and loser are in the graph (i.e., both are active)
        if winner_id in G.nodes and loser_id in G.nodes:
            edge_weights[(winner_id, loser_id)] += 1

    for (winner_id, loser_id), weight in edge_weights.items():
        G.add_edge(winner_id, loser_id, weight=weight)

    return G


def generate_graph_data_for_visualization(G):
    """
    Generate JSON-serializable data for D3.js force-directed graph.
    """
    nodes = []
    links = []

    # Create nodes
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        nodes.append({
            'id': str(node_id),
            'name': node_data['name'],
            'mu': node_data['mu'],
            'sigma_sq': node_data['sigma_sq'],
            'size': 5 + (node_data['mu'] + 3) * 2  # Scale by mu for visual sizing
        })

    # Create links
    for source, target in G.edges():
        weight = G[source][target].get('weight', 1)
        links.append({
            'source': str(source),
            'target': str(target),
            'weight': weight,
            'value': weight  # D3 uses 'value' for link strength
        })

    return {'nodes': nodes, 'links': links}


# ========================================
# 3. SCORE CONFIDENCE INTERVALS
# ========================================

# Two-sided 95%: crowd-BT keeps a Gaussian posterior over each project's
# score, so sigma_sq is the variance of mu and the interval is mu +/- z*sigma.
CONFIDENCE_Z = 1.96


def get_confidence_intervals(items=None, comparisons=None):
    """
    Every active project's score with its 95% interval, best first.

    Alongside each interval is the range of ranks the project could plausibly
    hold: its best rank counts only the projects whose whole interval sits
    above its own, and its worst counts every project whose interval reaches
    it. "#2-#7" says how settled a position is more directly than sigma_sq.
    """
    items, comparisons = _resolve(items, comparisons)
    items = _active(items)

    compared = defaultdict(int)
    for winner_id, loser_id, _annotator_id, _time in comparisons:
        compared[winner_id] += 1
        compared[loser_id] += 1

    rows = []
    for item in items:
        mu = float(item.mu)
        half = CONFIDENCE_Z * float(item.sigma_sq) ** 0.5
        rows.append({
            'id': item.id,
            'name': item.name,
            'location': item.location,
            'mu': mu,
            'low': mu - half,
            'high': mu + half,
            'comparisons': compared.get(item.id, 0),
        })
    rows.sort(key=lambda r: (-r['mu'], r['id']))

    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
        row['best_rank'] = 1 + sum(1 for o in rows if o['low'] > row['high'])
        row['worst_rank'] = sum(1 for o in rows if o['high'] >= row['low'])

    low = min((r['low'] for r in rows), default=-1.0)
    high = max((r['high'] for r in rows), default=1.0)
    return {'rows': rows, 'scale_low': low, 'scale_high': high}


# ========================================
# 3b. RANKING STABILITY
# ========================================

# Judging runs about an hour, at roughly eight votes a minute, so "recently"
# means the last ten minutes of voting.
STABILITY_RECENT_MINUTES = 10
STABILITY_TOP_NS = (3, 5, 10)


def replay_scores(comparisons, item_ids):
    """
    Re-run every vote through crowd-BT, yielding the scores after each one.

    Only current scores are stored, so this is how earlier rankings are
    recovered. It reproduces the stored scores exactly -- checked against the
    522 votes of Spring 26, to within 1e-15 -- because it applies the same
    update, from the same priors, in the order the votes were cast.
    """
    from gavel import crowd_bt

    mu = {i: crowd_bt.MU_PRIOR for i in item_ids}
    sigma_sq = {i: crowd_bt.SIGMA_SQ_PRIOR for i in item_ids}
    judges = {}
    for winner_id, loser_id, annotator_id, time in comparisons:
        if winner_id not in mu or loser_id not in mu:
            continue
        alpha, beta = judges.get(
            annotator_id, (crowd_bt.ALPHA_PRIOR, crowd_bt.BETA_PRIOR))
        (alpha, beta, mu[winner_id], sigma_sq[winner_id], mu[loser_id],
         sigma_sq[loser_id]) = crowd_bt.update(
            alpha, beta, mu[winner_id], sigma_sq[winner_id],
            mu[loser_id], sigma_sq[loser_id])
        judges[annotator_id] = (alpha, beta)
        yield time, mu


def get_ranking_stability(items=None, comparisons=None,
                          top_ns=STABILITY_TOP_NS,
                          recent_minutes=STABILITY_RECENT_MINUTES):
    """
    How long the top of the ranking has held still.

    For each N, two questions: has the *order* of the top N changed (who is
    first, second, third), and has the *set* changed (who is in the top N at
    all)? Each is answered with the number of votes, and minutes of voting,
    since it last moved, plus how many times it moved in the last
    `recent_minutes`. Minutes are measured to the latest vote rather than to
    now, so the numbers stop growing once judging ends.
    """
    items, comparisons = _resolve(items, comparisons)
    active_ids = [i.id for i in _active(items)]
    # A top N that covers every project is just the whole ranking.
    top_ns = [n for n in top_ns if n < len(active_ids)]
    if not comparisons or not top_ns:
        return {'rows': [], 'votes': len(comparisons),
                'recent_minutes': recent_minutes}

    all_ids = [i.id for i in items]
    last = {n: {'order': None, 'set': None} for n in top_ns}
    changed_at = {n: {'order': [], 'set': []} for n in top_ns}

    vote = 0
    last_time = None
    for time, mu in replay_scores(comparisons, all_ids):
        vote += 1
        last_time = time
        ranking = sorted(active_ids, key=lambda i: (-mu[i], i))
        for n in top_ns:
            order = tuple(ranking[:n])
            members = frozenset(order)
            if order != last[n]['order']:
                changed_at[n]['order'].append((vote, time))
                last[n]['order'] = order
            if members != last[n]['set']:
                changed_at[n]['set'].append((vote, time))
                last[n]['set'] = members

    recent_start = last_time - timedelta(minutes=recent_minutes)

    def summarize(changes):
        since_vote, since_time = changes[-1]
        return {
            'votes': vote - since_vote,
            'minutes': (last_time - since_time).total_seconds() / 60,
            'recent_changes': sum(1 for _v, t in changes if t > recent_start),
        }

    rows = [{'n': n,
             'order': summarize(changed_at[n]['order']),
             'set': summarize(changed_at[n]['set'])} for n in top_ns]
    return {'rows': rows, 'votes': vote, 'recent_minutes': recent_minutes}


# ========================================
# 3c. CONVERGENCE
# ========================================

# The top 3 is what gets announced, so convergence is judged on it.
CONVERGENCE_TOP_N = 3
# Held this long, in minutes and votes of voting, the top 3 counts as stable.
# Sized for an hour of judging: Spring 26's top-3 order last moved with 25
# minutes and 190 votes still to go.
STABLE_MINUTES = 10
STABLE_VOTES = 30


def get_convergence(confidence, stability, min_comparisons=None):
    """
    Whether the ranking has settled, as a status and the reason for it.

    This replaces a status read off the average sigma_sq, which falls with
    every vote whether or not the ranking is settling: Spring 26 read
    "Converged" from minute 35 while a top-3 project could still have ranked
    anywhere in 49 of 69 places. The statuses below are about the top of the
    ranking, which is what organizers announce:

      Not Started  no votes
      Collecting   some project has too few comparisons to place at all
      Settling     the top 3 has moved recently
      Stable       the top-3 order has held for STABLE_MINUTES and
                   STABLE_VOTES of voting
      Decided      the top 3 are statistically separated from everyone
                   else: none of them could rank below third
    """
    import gavel.settings as settings

    if min_comparisons is None:
        min_comparisons = settings.MIN_VIEWS
    rows = confidence['rows']
    if not rows or not stability['votes']:
        return {'status': 'Not Started', 'detail': 'No votes yet.'}

    short = [r for r in rows if r['comparisons'] < min_comparisons]
    if short:
        return {'status': 'Collecting', 'detail': (
            '%d of %d projects have fewer than %d comparisons.'
            % (len(short), len(rows), min_comparisons))}

    n = CONVERGENCE_TOP_N
    top = rows[:n]
    if len(rows) > n and all(r['worst_rank'] <= n for r in top):
        return {'status': 'Decided', 'detail': (
            'None of the top %d could rank lower than #%d.' % (n, n))}

    held = next((r['order'] for r in stability['rows'] if r['n'] == n), None)
    if held is None:
        # Too few projects for a top 3 to mean anything.
        return {'status': 'Settling', 'detail': 'Too few projects to rank a top %d.' % n}
    minutes = int(held['minutes'])
    if held['minutes'] >= STABLE_MINUTES and held['votes'] >= STABLE_VOTES:
        return {'status': 'Stable', 'detail': (
            'Top-%d order unchanged for %d votes (%d min), but its scores still '
            'overlap with projects below it.' % (n, held['votes'], minutes))}
    return {'status': 'Settling', 'detail': (
        'Top-%d order last changed %d vote%s ago (%d min); stable after %d '
        'min and %d votes.' % (n, held['votes'], '' if held['votes'] == 1 else 's',
                                minutes, STABLE_MINUTES, STABLE_VOTES))}


# ========================================
# 4. VOTING ACTIVITY TIMELINE
# ========================================

def get_voting_timeline(hours=2, comparisons=None):
    """
    Get voting activity over time.
    Returns vote counts in 15-second buckets for the last N hours.
    """
    if comparisons is None:
        comparisons = load_comparisons()
    decisions = comparisons

    if not decisions:
        return {
            'timeline_data': [],
            'total_votes': 0,
            'votes_last_minute': 0,
            'peak_time': None,
            'avg_votes_per_minute': 0
        }

    # Get time range
    now = datetime.utcnow()
    start_time = now - timedelta(hours=hours)

    # Filter decisions in time range
    recent_decisions = [d for d in decisions if d[3] >= start_time]

    # Create 15-second buckets
    bucket_counts = defaultdict(int)
    for dec in recent_decisions:
        # Round down to nearest 15 seconds
        bucket_time = dec[3].replace(microsecond=0)
        second = (bucket_time.second // 15) * 15
        bucket_time = bucket_time.replace(second=second)
        bucket_counts[bucket_time] += 1

    # Fill in missing buckets with 0
    timeline_data = []
    current = start_time.replace(microsecond=0)
    current = current.replace(second=(current.second // 15) * 15)

    while current <= now:
        count = bucket_counts.get(current, 0)
        timeline_data.append({
            'time': current.isoformat(),
            'timestamp': int(current.timestamp() * 1000),  # milliseconds for JS
            'display_time': current.strftime('%H:%M:%S'),
            'count': count
        })
        current += timedelta(seconds=15)

    # Calculate statistics
    total_votes = len(decisions)

    # Votes in last minute (4 buckets of 15 seconds)
    votes_last_minute = sum(bucket_counts.get(now.replace(microsecond=0).replace(second=(now.second // 15) * 15) - timedelta(seconds=15*i), 0) for i in range(4))

    peak_bucket = max(timeline_data, key=lambda x: x['count']) if timeline_data else None
    peak_time = peak_bucket['display_time'] if peak_bucket else None

    # Average votes per minute
    total_minutes = hours * 60
    avg_votes_per_minute = len(recent_decisions) / total_minutes if total_minutes > 0 else 0

    return {
        'timeline_data': timeline_data,
        'total_votes': total_votes,
        'votes_last_minute': votes_last_minute,
        'peak_time': peak_time,
        'avg_votes_per_minute': avg_votes_per_minute
    }


# ========================================
# 7. STATISTICAL SUMMARY DASHBOARD
# ========================================

def get_statistical_summary(items=None, judges=None, comparisons=None):
    """
    Calculate comprehensive statistical summary for dashboard.
    """
    items, comparisons = _resolve(items, comparisons)
    items = _active(items)
    if judges is None:
        judges = Annotator.query_current().all()
    decisions = comparisons

    if not items or not decisions:
        return {
            'total_comparisons': 0,
            'total_projects': len(items),
            'total_judges': len(judges),
            'avg_comparisons_per_project': 0,
            'avg_comparisons_per_judge': 0,
        }

    # Basic counts
    total_comparisons = len(decisions)
    total_projects = len(items)
    total_judges = len(judges)
    active_judges = len([j for j in judges if j.active])

    # Comparisons per project
    project_comparison_counts = defaultdict(int)
    for winner_id, loser_id, _annotator_id, _time in decisions:
        project_comparison_counts[winner_id] += 1
        project_comparison_counts[loser_id] += 1

    avg_comparisons_per_project = sum(project_comparison_counts.values()) / (2 * total_projects) if total_projects > 0 else 0

    # Comparisons per judge
    judge_comparison_counts = defaultdict(int)
    for _winner_id, _loser_id, annotator_id, _time in decisions:
        judge_comparison_counts[annotator_id] += 1

    avg_comparisons_per_judge = total_comparisons / total_judges if total_judges > 0 else 0

    # Projects with high uncertainty
    high_uncertainty_projects = sorted(
        [(item.name, float(item.sigma_sq)) for item in items],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        'total_comparisons': total_comparisons,
        'total_projects': total_projects,
        'total_judges': total_judges,
        'active_judges': active_judges,
        'avg_comparisons_per_project': round(avg_comparisons_per_project, 1),
        'avg_comparisons_per_judge': round(avg_comparisons_per_judge, 1),
        'high_uncertainty_projects': high_uncertainty_projects
    }
