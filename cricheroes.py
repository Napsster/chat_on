"""Read one CricHeroes scorecard page into a match result.

CricHeroes has no public API. Its scorecard pages are Next.js and embed the
match as JSON inside `self.__next_f.push([1, "..."])` scripts; the resolved
copy sits under "summaryData":{"status":true,"data":{...}}. If CricHeroes
changes that layout, parse_page() raises ValueError and HR falls back to
typing the result by hand in the admin page.
"""
import json
import re
import urllib.request

_URL = re.compile(r"^https?://(?:www\.)?cricheroes\.(?:com|in)/scorecard/(\d+)(?:/([^/?#]+))?(?:/([^/?#]+))?")


def match_id_from_url(url: str) -> str:
    m = _URL.match((url or "").strip())
    if not m:
        raise ValueError("Not a CricHeroes scorecard link (expected cricheroes.com/scorecard/<id>/...)")
    return m.group(1)


def _get(page: str, timeout: int) -> str:
    req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch(url: str, timeout: int = 20) -> dict:
    url = url.strip()
    mid = match_id_from_url(url)
    m = _URL.match(url)
    # Ask for the scorecard view (the slugs are cosmetic). Before the match
    # that view is empty, so fall back to the link as pasted — the /upcoming
    # page — which parse_page() recognises as "hasn't started yet".
    page = f"https://cricheroes.com/scorecard/{mid}/{m.group(2) or 'x'}/{m.group(3) or 'y'}/scorecard"
    try:
        result = parse_page(_get(page, timeout))
    except ValueError as e:
        if "hasn't started" in str(e) or page == url:
            raise
        result = parse_page(_get(url, timeout))
    result["url"] = url
    return result


def parse_page(html: str) -> dict:
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.S)
    blob = "".join(json.loads('"' + c + '"') for c in chunks)
    key = '"summaryData":{"status":true'
    i = blob.find(key)
    if i < 0:
        if '"upcomingMatchDetails":{"status":true' in blob:
            raise ValueError("This match hasn't started on CricHeroes yet (no toss recorded)")
        raise ValueError("Could not find match data on the CricHeroes page")
    data, _ = json.JSONDecoder().raw_decode(blob, i + len('"summaryData":'))
    d = data["data"]
    # A live match is saved too (toss + score so far), marked in progress;
    # saving the same link after the match replaces it with the final result.

    def team(t):
        inn = (t.get("innings") or [{}])[-1].get("summary") or {}
        return {"name": t.get("name", ""), "score": t.get("summary", ""), "overs": re.sub(r"[()]|\s*Ov\b", "", inn.get("over", "")).strip()}

    best = d.get("best_performances") or {}
    batting = sorted(best.get("batting") or [], key=lambda b: (-int(b.get("runs") or 0), int(b.get("balls") or 0)))
    bowling = sorted(best.get("bowling") or [],
                     key=lambda b: (-int(b.get("wickets") or 0), int(b.get("runs") or 0)))
    potm = d.get("player_of_the_match") or {}
    return {
        "match_id": str(d.get("match_id", "")),
        "final": d.get("status") == "past",
        "date": (d.get("start_datetime") or "")[:10],
        "overs": d.get("overs"),
        "toss": (d.get("toss_details") or "").removeprefix("Toss:").strip(),
        "result": (d.get("match_summary") or {}).get("summary") or "",
        "teams": [team(d.get("team_a") or {}), team(d.get("team_b") or {})],
        "player_of_the_match": f"{potm['player_name']} ({potm.get('team_name', '')})" if potm.get("player_name") else "",
        "top_batters": [f"{b['player_name']} ({b.get('team_name', '')}) {b.get('runs')} off {b.get('balls')}"
                        for b in batting[:3]],
        "top_bowlers": [f"{b['player_name']} ({b.get('team_name', '')}) {b.get('wickets')}/{b.get('runs')} in {b.get('overs')} ov"
                        for b in bowling[:3]],
    }


def summary_text(r: dict) -> str:
    """One readable block per match, used both in the admin list and for Buddy."""
    if r.get("text"):  # typed by hand
        return r["text"].strip()
    a, b = (r.get("teams") or [{}, {}])[:2]
    def side(t):
        return f"{t.get('name')} {t.get('score')} ({t.get('overs')} ov)" if t.get("score") else f"{t.get('name')} (yet to bat)"
    lines = [("IN PROGRESS, NOT FINAL: " if r.get("final") is False else "")
             + f"{side(a)} vs {side(b)}" + (f" — {r['date']}" if r.get("date") else "")]
    if r.get("result"):
        lines.append(f"{'Result' if r.get('final') is not False else 'Status'}: {r['result']}")
    if r.get("toss"):
        lines.append(f"Toss: {r['toss']}")
    if r.get("player_of_the_match"):
        lines.append(f"Player of the match: {r['player_of_the_match']}")
    if r.get("top_batters"):
        lines.append("Top batters: " + "; ".join(r["top_batters"]))
    if r.get("top_bowlers"):
        lines.append("Top bowlers: " + "; ".join(r["top_bowlers"]))
    return "\n".join(lines)
