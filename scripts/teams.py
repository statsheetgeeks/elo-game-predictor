"""
Static MLB team metadata, keyed by the exact team "name" string returned by
the MLB Stats API (statsapi.mlb.com) — the same string used as home_team /
away_team throughout the elo pipeline. Used by update_data.py to attach a
logo URL and brand colors to every team row/JSON record.

Logo URLs follow MLB's own public static-asset pattern
(https://www.mlbstatic.com/team-logos/{team_id}.svg) — official MLB assets,
not redistributed copies.
"""

TEAMS = {
    "Arizona Diamondbacks":  {"id": 109, "abbr": "ARI", "primary": "#A71930", "secondary": "#E3D4AD"},
    "Atlanta Braves":        {"id": 144, "abbr": "ATL", "primary": "#CE1141", "secondary": "#13274F"},
    "Baltimore Orioles":     {"id": 110, "abbr": "BAL", "primary": "#DF4601", "secondary": "#000000"},
    "Boston Red Sox":        {"id": 111, "abbr": "BOS", "primary": "#BD3039", "secondary": "#0C2340"},
    "Chicago Cubs":          {"id": 112, "abbr": "CHC", "primary": "#0E3386", "secondary": "#CC3433"},
    "Chicago White Sox":     {"id": 145, "abbr": "CWS", "primary": "#27251F", "secondary": "#C4CED4"},
    "Cincinnati Reds":       {"id": 113, "abbr": "CIN", "primary": "#C6011F", "secondary": "#000000"},
    "Cleveland Guardians":   {"id": 114, "abbr": "CLE", "primary": "#00385D", "secondary": "#E50022"},
    "Colorado Rockies":      {"id": 115, "abbr": "COL", "primary": "#333366", "secondary": "#C4CED4"},
    "Detroit Tigers":        {"id": 116, "abbr": "DET", "primary": "#0C2340", "secondary": "#FA4616"},
    "Houston Astros":        {"id": 117, "abbr": "HOU", "primary": "#002D62", "secondary": "#EB6E1F"},
    "Kansas City Royals":    {"id": 118, "abbr": "KC",  "primary": "#004687", "secondary": "#BD9B60"},
    "Los Angeles Angels":    {"id": 108, "abbr": "LAA", "primary": "#BA0021", "secondary": "#003263"},
    "Los Angeles Dodgers":   {"id": 119, "abbr": "LAD", "primary": "#005A9C", "secondary": "#EF3E42"},
    "Miami Marlins":         {"id": 146, "abbr": "MIA", "primary": "#00A3E0", "secondary": "#EF3340"},
    "Milwaukee Brewers":     {"id": 158, "abbr": "MIL", "primary": "#12284B", "secondary": "#FFC52F"},
    "Minnesota Twins":       {"id": 142, "abbr": "MIN", "primary": "#002B5C", "secondary": "#D31145"},
    "New York Mets":         {"id": 121, "abbr": "NYM", "primary": "#002D72", "secondary": "#FF5910"},
    "New York Yankees":      {"id": 147, "abbr": "NYY", "primary": "#003087", "secondary": "#E4002C"},
    "Athletics":             {"id": 133, "abbr": "ATH", "primary": "#003831", "secondary": "#EFB21E"},
    "Philadelphia Phillies": {"id": 143, "abbr": "PHI", "primary": "#E81828", "secondary": "#002D72"},
    "Pittsburgh Pirates":    {"id": 134, "abbr": "PIT", "primary": "#FDB827", "secondary": "#27251F"},
    "San Diego Padres":      {"id": 135, "abbr": "SD",  "primary": "#2F241D", "secondary": "#FFC425"},
    "San Francisco Giants":  {"id": 137, "abbr": "SF",  "primary": "#FD5A1E", "secondary": "#27251F"},
    "Seattle Mariners":      {"id": 136, "abbr": "SEA", "primary": "#0C2C56", "secondary": "#005C5C"},
    "St. Louis Cardinals":   {"id": 138, "abbr": "STL", "primary": "#C41E3A", "secondary": "#0C2340"},
    "Tampa Bay Rays":        {"id": 139, "abbr": "TB",  "primary": "#092C5C", "secondary": "#8FBCE6"},
    "Texas Rangers":         {"id": 140, "abbr": "TEX", "primary": "#003278", "secondary": "#C0111F"},
    "Toronto Blue Jays":     {"id": 141, "abbr": "TOR", "primary": "#134A8E", "secondary": "#E8291C"},
    "Washington Nationals":  {"id": 120, "abbr": "WSH", "primary": "#AB0003", "secondary": "#14225A"},
}

# A couple of alternate names the Stats API has used historically (renames,
# relocations) so lookups don't silently fail if an older cached season used
# a different string.
ALIASES = {
    "Oakland Athletics": "Athletics",
    "Cleveland Indians": "Cleveland Guardians",
}


def team_meta(name: str) -> dict:
    """Look up a team's metadata by its MLB Stats API name, with a safe
    fallback for any team not in the table (keeps the pipeline from
    crashing on a name we haven't seen yet)."""
    canonical = ALIASES.get(name, name)
    meta = TEAMS.get(canonical)
    if meta is None:
        return {"id": None, "abbr": name[:3].upper(), "primary": "#4B5563", "secondary": "#9CA3AF"}
    return meta


def logo_url(name: str) -> str:
    meta = team_meta(name)
    if meta["id"] is None:
        return ""
    return f"https://www.mlbstatic.com/team-logos/{meta['id']}.svg"


def teams_json_blob() -> dict:
    """Full lookup table for the frontend: name -> {abbr, primary, secondary, logo}."""
    out = {}
    for name in TEAMS:
        meta = team_meta(name)
        out[name] = {
            "abbr": meta["abbr"],
            "primary": meta["primary"],
            "secondary": meta["secondary"],
            "logo": logo_url(name),
        }
    return out
