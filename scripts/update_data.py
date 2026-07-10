"""
Headless daily data pipeline for the MLB Elo site.

This is a plain-script port of the original marimo notebook: same elo
model, same walk-forward auto-tuning, same CSV outputs -- but with the
interactive/UI cells stripped out (this runs unattended in GitHub Actions)
and three additions:

  1. A join of today's probable starters against season pitcher ratings
     (mlb_daily_starting_pitchers_{date}.csv) -- the notebook only ever
     produced *season-wide* pitcher ratings, not "who's starting today".
  2. A compact docs/data/latest.json bundle so the frontend does one fetch
     instead of parsing five CSVs client-side.
  3. An append-only predictions log + pre-aggregated summary for the three
     trackers at the bottom of the page (combined-elo record, home-prob
     record, calibration-by-bucket).

Run with: python scripts/update_data.py
"""
from __future__ import annotations

import json
import pathlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import differential_evolution, minimize_scalar

from teams import team_meta, logo_url, teams_json_blob

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ELO_DIR = DATA_DIR / "raw" / "elo"
DOCS_DATA_DIR = REPO_ROOT / "docs" / "data"
CSV_DIR = DOCS_DATA_DIR / "csv"
PREDICTIONS_LOG_PATH = DOCS_DATA_DIR / "predictions_log.csv"

for _d in (CACHE_DIR, ELO_DIR, DOCS_DATA_DIR, CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TODAY = date.today()

# ---------------------------------------------------------------------
# Fixed data-scope constants (unchanged from the notebook)
# ---------------------------------------------------------------------
START_SEASON = TODAY.year - 6
END_SEASON = TODAY.year
REFETCH_WINDOW_DAYS = 4
INIT_RATING = 1500.0

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LOGIT_TO_ELO = 400.0 / np.log(10.0)


# =======================================================================
# Schedule fetch + cache (identical logic to the notebook)
# =======================================================================
def fetch_schedule_range(start_date: str, end_date: str, season: int) -> pd.DataFrame:
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
        "hydrate": "probablePitcher",
    }
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            teams = g["teams"]
            home, away = teams["home"], teams["away"]
            rows.append(
                {
                    "game_pk": g["gamePk"],
                    "date": day["date"],
                    "year": season,
                    "status": g.get("status", {}).get("abstractGameState"),
                    "home_team": home["team"]["name"],
                    "away_team": away["team"]["name"],
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "home_probable_pitcher": (home.get("probablePitcher") or {}).get("fullName"),
                    "away_probable_pitcher": (away.get("probablePitcher") or {}).get("fullName"),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    return out


def get_season_schedule(season: int, cache_dir: pathlib.Path, refetch_window_days: int = 4):
    cache_path = cache_dir / f"schedule_{season}.csv"
    today = TODAY
    is_current_season = season >= today.year

    cached = None
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
        except (pd.errors.EmptyDataError, ValueError):
            cached = None
        if cached is not None and cached.empty:
            cached = None

    if cached is not None:
        if not is_current_season:
            return cached, "cache (complete past season, no pull)"

        completed = cached.dropna(subset=["home_score", "away_score"])
        if completed.empty:
            anchor = today
        else:
            anchor = min(completed["date"].max().date(), today)
        refetch_start = anchor - timedelta(days=refetch_window_days)
        fresh = fetch_schedule_range(refetch_start.isoformat(), today.isoformat(), season)

        if fresh.empty:
            return cached, "cache (no new games in window)"

        cached = cached[cached["date"] < pd.Timestamp(refetch_start)]
        merged = (
            pd.concat([cached, fresh], ignore_index=True)
            .drop_duplicates(subset="game_pk", keep="last")
            .sort_values(["date", "game_pk"])
            .reset_index(drop=True)
        )
        merged.to_csv(cache_path, index=False)
        return merged, f"refreshed trailing {refetch_window_days}-day window"

    full = fetch_schedule_range(f"{season}-01-01", f"{season}-12-31", season)
    full.to_csv(cache_path, index=False)
    return full, "full season pull (new cache)"


# =======================================================================
# Elo model (identical logic to the notebook)
# =======================================================================
def elo_prob(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def estimate_team_home_adv(elo_df, global_home_adv, lam):
    team_adv = {}
    for team, g in elo_df.groupby("home_team"):
        baseline_prob = elo_prob(g["home_elo_pre"] + global_home_adv, g["away_elo_pre"])
        resid_prob = (g["home_win"] - baseline_prob).mean()
        p_bar = min(max(baseline_prob.mean(), 0.05), 0.95)
        raw_logit_shift = resid_prob / (p_bar * (1 - p_bar))
        raw_elo_shift = raw_logit_shift * LOGIT_TO_ELO
        n = len(g)
        team_adv[team] = global_home_adv + raw_elo_shift * n / (n + lam)
    return team_adv


def build_elo(
    df,
    K=4,
    HOME_ADV=24,
    INIT_RATING=1500,
    SEASON_REVERT=1 / 3,
    use_mov=False,
    K1_MARGIN=1.0,
    SIGMA1_MARGIN=175.0,
    use_pitcher_adj=False,
    PITCHER_K=2.0,
    PITCHER_SIGMA1=175.0,
    PITCHER_WEIGHT=0.3,
    PITCHER_INIT=1500,
    PITCHER_SEASON_REVERT=0.5,
):
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    if df.empty:
        return df, {}
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    has_pitcher_cols = "home_probable_pitcher" in df.columns and "away_probable_pitcher" in df.columns

    def _home_adv_for(team):
        if isinstance(HOME_ADV, dict):
            return HOME_ADV.get(team, HOME_ADV.get("_default", 24))
        return HOME_ADV

    ratings = {}
    pitcher_ratings = {}
    cur_season = None
    rows = []
    all_teams = set(df["home_team"]) | set(df["away_team"])

    for _game in df.itertuples(index=False):
        game = _game._asdict()
        home, away, season = game["home_team"], game["away_team"], game["year"]

        if season != cur_season:
            cur_season = season
            for team in all_teams:
                if team in ratings:
                    ratings[team] = ratings[team] * (1 - SEASON_REVERT) + INIT_RATING * SEASON_REVERT
                else:
                    ratings[team] = INIT_RATING
            if use_pitcher_adj:
                for p in list(pitcher_ratings.keys()):
                    pitcher_ratings[p] = (
                        pitcher_ratings[p] * (1 - PITCHER_SEASON_REVERT) + PITCHER_INIT * PITCHER_SEASON_REVERT
                    )

        ratings.setdefault(home, INIT_RATING)
        ratings.setdefault(away, INIT_RATING)

        home_r_pre, away_r_pre = ratings[home], ratings[away]

        home_eff = home_r_pre + _home_adv_for(home)
        away_eff = away_r_pre

        home_pitcher = game.get("home_probable_pitcher") if has_pitcher_cols else None
        away_pitcher = game.get("away_probable_pitcher") if has_pitcher_cols else None
        hp_pre = ap_pre = None
        if use_pitcher_adj:
            if pd.notna(home_pitcher):
                hp_pre = pitcher_ratings.setdefault(home_pitcher, PITCHER_INIT)
                away_eff = away_eff - PITCHER_WEIGHT * (hp_pre - PITCHER_INIT)
            if pd.notna(away_pitcher):
                ap_pre = pitcher_ratings.setdefault(away_pitcher, PITCHER_INIT)
                home_eff = home_eff - PITCHER_WEIGHT * (ap_pre - PITCHER_INIT)

        home_elo_prob = elo_prob(home_eff, away_eff)
        home_win = int(game["home_score"] > game["away_score"])
        eff_diff = home_eff - away_eff

        if use_mov:
            margin = game["home_score"] - game["away_score"]
            margin_hat = eff_diff / SIGMA1_MARGIN
            delta = K1_MARGIN * (margin - margin_hat) + K * (home_win - home_elo_prob)
        else:
            delta = K * (home_win - home_elo_prob)

        new_home = home_r_pre + delta
        new_away = away_r_pre - delta
        ratings[home], ratings[away] = new_home, new_away

        if use_pitcher_adj:
            league_avg_runs = 4.5
            if pd.notna(home_pitcher):
                opp_strength = away_r_pre - INIT_RATING
                expected_runs_allowed = league_avg_runs + opp_strength / PITCHER_SIGMA1
                perf_margin = expected_runs_allowed - game["away_score"]
                pitcher_ratings[home_pitcher] = hp_pre + PITCHER_K * perf_margin
            if pd.notna(away_pitcher):
                opp_strength = home_r_pre - INIT_RATING
                expected_runs_allowed = league_avg_runs + opp_strength / PITCHER_SIGMA1
                perf_margin = expected_runs_allowed - game["home_score"]
                pitcher_ratings[away_pitcher] = ap_pre + PITCHER_K * perf_margin

        rows.append(
            {
                "date": game["date"],
                "season": season,
                "game_pk": game["game_pk"],
                "home_team": home,
                "away_team": away,
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "home_elo_pre": round(home_r_pre, 1),
                "away_elo_pre": round(away_r_pre, 1),
                "elo_diff": round(eff_diff, 1),
                "home_elo_prob": round(home_elo_prob, 4),
                "away_elo_prob": round(1 - home_elo_prob, 4),
                "home_elo_post": round(new_home, 1),
                "away_elo_post": round(new_away, 1),
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "home_win": home_win,
            }
        )

    return pd.DataFrame(rows), pitcher_ratings


def log_loss(elo_df, mask):
    p = elo_df.loc[mask, "home_elo_prob"].clip(1e-6, 1 - 1e-6)
    y = elo_df.loc[mask, "home_win"]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def auto_season_split(seasons_sorted):
    n = len(seasons_sorted)
    if n < 3:
        return 0, 0, seasons_sorted, []
    test_n = max(1, round(n * 0.2))
    burn_in_n = max(1, round(n * 0.3))
    while burn_in_n + test_n >= n:
        if burn_in_n > 1:
            burn_in_n -= 1
        elif test_n > 1:
            test_n -= 1
        else:
            break
    val_seasons = seasons_sorted[burn_in_n : n - test_n]
    test_seasons = seasons_sorted[n - test_n :]
    return burn_in_n, test_n, val_seasons, test_seasons


def main():
    print(f"=== MLB elo pipeline run: {TODAY.isoformat()} ===")

    # -------------------------------------------------------------
    # 1. Pull schedule for all loaded seasons
    # -------------------------------------------------------------
    frames = []
    for season in range(START_SEASON, END_SEASON + 1):
        df, status = get_season_schedule(season, CACHE_DIR, REFETCH_WINDOW_DAYS)
        frames.append(df)
        print(f"  {season}: {status} ({len(df)} games)")
    schedule_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if schedule_df.empty:
        print("No schedule data pulled -- aborting.")
        return

    all_seasons_sorted = sorted(schedule_df["year"].unique())
    burn_in_n, test_n, val_seasons, test_seasons = auto_season_split(all_seasons_sorted)

    # -------------------------------------------------------------
    # 2. Stage 1: base params (K / home_adv / season revert)
    # -------------------------------------------------------------
    if not val_seasons:
        base_K, base_HOME_ADV, base_SEASON_REVERT = 4.0, 24.0, 1 / 3
        base_val_ll = float("nan")
    else:
        def _base_objective(x):
            K_, HA_, REV_ = x
            elo_, _ = build_elo(schedule_df, K=K_, HOME_ADV=HA_, INIT_RATING=INIT_RATING, SEASON_REVERT=REV_)
            mask = elo_["season"].isin(val_seasons)
            return log_loss(elo_, mask) if mask.sum() else 10.0

        res = differential_evolution(
            _base_objective, bounds=[(1, 40), (0, 80), (0, 1)],
            maxiter=8, popsize=8, tol=1e-4, seed=42, polish=True, workers=1,
        )
        base_K, base_HOME_ADV, base_SEASON_REVERT = (float(v) for v in res.x)
        base_val_ll = float(res.fun)
    print(f"  Stage 1 base: K={base_K:.2f} home_adv={base_HOME_ADV:.2f} revert={base_SEASON_REVERT:.3f}")

    # -------------------------------------------------------------
    # 3. Stage 2: margin of victory
    # -------------------------------------------------------------
    if not val_seasons:
        use_mov = False
        K1_MARGIN, SIGMA1_MARGIN = 1.0, 175.0
        mov_val_ll = base_val_ll
    else:
        def _mov_objective(x):
            K1_, S1_ = x
            elo_, _ = build_elo(
                schedule_df, K=base_K, HOME_ADV=base_HOME_ADV, INIT_RATING=INIT_RATING,
                SEASON_REVERT=base_SEASON_REVERT, use_mov=True, K1_MARGIN=K1_, SIGMA1_MARGIN=S1_,
            )
            mask = elo_["season"].isin(val_seasons)
            return log_loss(elo_, mask) if mask.sum() else 10.0

        res = differential_evolution(
            _mov_objective, bounds=[(0, 5), (25, 500)], maxiter=8, popsize=8, tol=1e-4, seed=42, polish=True, workers=1,
        )
        K1_MARGIN, SIGMA1_MARGIN = (float(v) for v in res.x)
        mov_ll = float(res.fun)
        use_mov = mov_ll < base_val_ll
        mov_val_ll = mov_ll if use_mov else base_val_ll
    print(f"  Stage 2 MOV: {'on' if use_mov else 'off'}")

    # -------------------------------------------------------------
    # 4. Stage 3: starting-pitcher layer
    # -------------------------------------------------------------
    if not val_seasons or "home_probable_pitcher" not in schedule_df.columns:
        use_pitcher_adj = False
        PITCHER_K, PITCHER_SIGMA1, PITCHER_WEIGHT, PITCHER_SEASON_REVERT = 2.0, 175.0, 0.3, 0.5
        pitcher_val_ll = mov_val_ll
    else:
        def _pitcher_objective(x):
            PK_, PS_, PW_, PREV_ = x
            elo_, _ = build_elo(
                schedule_df, K=base_K, HOME_ADV=base_HOME_ADV, INIT_RATING=INIT_RATING,
                SEASON_REVERT=base_SEASON_REVERT, use_mov=use_mov, K1_MARGIN=K1_MARGIN, SIGMA1_MARGIN=SIGMA1_MARGIN,
                use_pitcher_adj=True, PITCHER_K=PK_, PITCHER_SIGMA1=PS_, PITCHER_WEIGHT=PW_,
                PITCHER_INIT=INIT_RATING, PITCHER_SEASON_REVERT=PREV_,
            )
            mask = elo_["season"].isin(val_seasons)
            return log_loss(elo_, mask) if mask.sum() else 10.0

        res = differential_evolution(
            _pitcher_objective, bounds=[(0, 10), (25, 500), (0, 1), (0, 1)],
            maxiter=8, popsize=6, tol=1e-4, seed=42, polish=True, workers=1,
        )
        PITCHER_K, PITCHER_SIGMA1, PITCHER_WEIGHT, PITCHER_SEASON_REVERT = (float(v) for v in res.x)
        p_ll = float(res.fun)
        use_pitcher_adj = p_ll < mov_val_ll
        pitcher_val_ll = p_ll if use_pitcher_adj else mov_val_ll
    print(f"  Stage 3 pitcher adj: {'on' if use_pitcher_adj else 'off'}")

    # -------------------------------------------------------------
    # 5. Stage 4: team/park home advantage
    # -------------------------------------------------------------
    if not val_seasons or burn_in_n == 0:
        use_team_home_adv = False
        team_home_adv = {}
    else:
        common = dict(
            K=base_K, INIT_RATING=INIT_RATING, SEASON_REVERT=base_SEASON_REVERT,
            use_mov=use_mov, K1_MARGIN=K1_MARGIN, SIGMA1_MARGIN=SIGMA1_MARGIN,
            use_pitcher_adj=use_pitcher_adj, PITCHER_K=PITCHER_K, PITCHER_SIGMA1=PITCHER_SIGMA1,
            PITCHER_WEIGHT=PITCHER_WEIGHT, PITCHER_INIT=INIT_RATING, PITCHER_SEASON_REVERT=PITCHER_SEASON_REVERT,
        )
        pass1, _ = build_elo(schedule_df, HOME_ADV=base_HOME_ADV, **common)
        burn_in_seasons = all_seasons_sorted[:burn_in_n]
        pre_val = pass1[pass1["season"].isin(burn_in_seasons)]

        if pre_val.empty:
            use_team_home_adv = False
            team_home_adv = {}
        else:
            def _lambda_objective(lam):
                adv = estimate_team_home_adv(pre_val, base_HOME_ADV, lam)
                adv["_default"] = base_HOME_ADV
                elo_, _ = build_elo(schedule_df, HOME_ADV=adv, **common)
                mask = elo_["season"].isin(val_seasons)
                return log_loss(elo_, mask) if mask.sum() else 10.0

            res = minimize_scalar(_lambda_objective, bounds=(0, 1000), method="bounded", options={"xatol": 1.0})
            team_home_adv_lambda = float(res.x)
            ha_ll = float(res.fun)
            use_team_home_adv = ha_ll < pitcher_val_ll
            team_home_adv = estimate_team_home_adv(pre_val, base_HOME_ADV, team_home_adv_lambda)
            team_home_adv["_default"] = base_HOME_ADV
    print(f"  Stage 4 team home adv: {'on' if use_team_home_adv else 'off'}")

    # -------------------------------------------------------------
    # 6. Final build with everything the optimizer chose
    # -------------------------------------------------------------
    common_kwargs = dict(
        K=base_K, INIT_RATING=INIT_RATING, SEASON_REVERT=base_SEASON_REVERT,
        use_mov=use_mov, K1_MARGIN=K1_MARGIN, SIGMA1_MARGIN=SIGMA1_MARGIN,
        use_pitcher_adj=use_pitcher_adj, PITCHER_K=PITCHER_K, PITCHER_SIGMA1=PITCHER_SIGMA1,
        PITCHER_WEIGHT=PITCHER_WEIGHT, PITCHER_INIT=INIT_RATING, PITCHER_SEASON_REVERT=PITCHER_SEASON_REVERT,
    )
    home_adv_param = team_home_adv if (use_team_home_adv and team_home_adv) else base_HOME_ADV
    elo_df, pitcher_ratings = build_elo(schedule_df, HOME_ADV=home_adv_param, **common_kwargs)

    if elo_df.empty:
        print("Elo build produced no rows -- aborting.")
        return

    elo_df.to_csv(ELO_DIR / "mlb_elo_auto.csv", index=False)

    # -------------------------------------------------------------
    # 7. Today's season rankings
    # -------------------------------------------------------------
    latest_season = elo_df["season"].max()
    cur = elo_df[elo_df["season"] == latest_season]
    home_r = cur.groupby("home_team")["home_elo_post"].last()
    away_r = cur.groupby("away_team")["away_elo_post"].last()
    latest = pd.concat([home_r, away_r]).groupby(level=0).last().sort_values(ascending=False)

    rankings_df = latest.reset_index()
    rankings_df.columns = ["team", "elo_rating"]
    league_avg_rating = latest.mean()
    rankings_df["anticipated_win_pct"] = (
        100.0 / (1.0 + 10 ** ((league_avg_rating - rankings_df["elo_rating"]) / 400.0))
    ).round(1)
    rankings_df.insert(0, "rank", range(1, len(rankings_df) + 1))
    rankings_df.insert(0, "as_of_date", TODAY.isoformat())
    rankings_df.to_csv(CSV_DIR / f"mlb_elo_rankings_{TODAY.isoformat()}.csv", index=False)

    # -------------------------------------------------------------
    # 8. Pitcher ratings (season-wide, all starters)
    # -------------------------------------------------------------
    if use_pitcher_adj and pitcher_ratings:
        pitcher_ratings_df = (
            pd.DataFrame([{"pitcher": p, "rating": round(r, 1)} for p, r in pitcher_ratings.items()])
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
        pitcher_ratings_df.insert(0, "rank", range(1, len(pitcher_ratings_df) + 1))
        pitcher_ratings_df.insert(0, "as_of_date", TODAY.isoformat())
    else:
        pitcher_ratings_df = pd.DataFrame(columns=["as_of_date", "rank", "pitcher", "rating"])
    if not pitcher_ratings_df.empty:
        pitcher_ratings_df.to_csv(CSV_DIR / f"mlb_pitcher_ratings_{TODAY.isoformat()}.csv", index=False)

    # -------------------------------------------------------------
    # 9. Team/park home advantage
    # -------------------------------------------------------------
    if use_team_home_adv and team_home_adv:
        team_home_adv_df = (
            pd.DataFrame([{"team": t, "home_adv": round(v, 1)} for t, v in team_home_adv.items() if t != "_default"])
            .sort_values("home_adv", ascending=False)
            .reset_index(drop=True)
        )
        team_home_adv_df.insert(0, "as_of_date", TODAY.isoformat())
        team_home_adv_df.to_csv(CSV_DIR / f"mlb_team_home_adv_{TODAY.isoformat()}.csv", index=False)
    else:
        team_home_adv_df = pd.DataFrame(columns=["as_of_date", "team", "home_adv"])

    # -------------------------------------------------------------
    # 10. Combined rankings (elo + rotation pitcher effect + home_adv)
    # -------------------------------------------------------------
    def _team_rotation_avg(elo_df, pitcher_ratings):
        if not pitcher_ratings or elo_df.empty:
            return {}
        latest_season_ = elo_df["season"].max()
        cur_ = elo_df[elo_df["season"] == latest_season_]
        pieces = []
        for team_col, pitcher_col in (("home_team", "home_pitcher"), ("away_team", "away_pitcher")):
            sub = cur_[[team_col, pitcher_col]].dropna().rename(columns={team_col: "team", pitcher_col: "pitcher"})
            pieces.append(sub)
        starts_df = pd.concat(pieces, ignore_index=True)
        starts_df["rating"] = starts_df["pitcher"].map(pitcher_ratings)
        starts_df = starts_df.dropna(subset=["rating"])
        if starts_df.empty:
            return {}
        return starts_df.groupby("team")["rating"].mean().to_dict()

    combined_rankings_df = rankings_df.rename(columns={"rank": "elo_rank"}).copy()

    if use_team_home_adv and team_home_adv:
        combined_rankings_df["home_adv"] = combined_rankings_df["team"].map(
            lambda t: team_home_adv.get(t, team_home_adv.get("_default", base_HOME_ADV))
        ).round(1)
    else:
        combined_rankings_df["home_adv"] = round(base_HOME_ADV, 1)

    rotation_avg = _team_rotation_avg(elo_df, pitcher_ratings) if use_pitcher_adj else {}
    if use_pitcher_adj and rotation_avg:
        combined_rankings_df["avg_rotation_pitcher_rating"] = combined_rankings_df["team"].map(rotation_avg).round(1)
        combined_rankings_df["pitcher_rating_effect"] = (
            PITCHER_WEIGHT * (combined_rankings_df["avg_rotation_pitcher_rating"] - INIT_RATING)
        ).round(1)
        combined_rankings_df["pitcher_rating_effect"] = combined_rankings_df["pitcher_rating_effect"].fillna(0.0)
        combined_rankings_df["combined_power_rating"] = (
            combined_rankings_df["elo_rating"] + combined_rankings_df["pitcher_rating_effect"]
        ).round(1)
    else:
        combined_rankings_df["avg_rotation_pitcher_rating"] = float("nan")
        combined_rankings_df["pitcher_rating_effect"] = float("nan")
        combined_rankings_df["combined_power_rating"] = combined_rankings_df["elo_rating"]

    combined_rankings_df = combined_rankings_df.sort_values("combined_power_rating", ascending=False).reset_index(drop=True)
    combined_rankings_df.insert(0, "combined_rank", range(1, len(combined_rankings_df) + 1))
    combined_rankings_df = combined_rankings_df[
        ["as_of_date", "combined_rank", "team", "combined_power_rating", "elo_rating", "elo_rank",
         "home_adv", "avg_rotation_pitcher_rating", "pitcher_rating_effect"]
    ]
    combined_rankings_df.to_csv(CSV_DIR / f"mlb_combined_rankings_{TODAY.isoformat()}.csv", index=False)

    # -------------------------------------------------------------
    # 11. Today's matchups
    # -------------------------------------------------------------
    matchup_cols = [
        "as_of_date", "game_pk", "home_team", "home_combined_rating", "away_team",
        "away_combined_rating", "elo_diff", "home_win_prob",
    ]
    todays_games = schedule_df[schedule_df["date"].dt.date == TODAY]
    if todays_games.empty or combined_rankings_df.empty:
        daily_matchups_df = pd.DataFrame(columns=matchup_cols)
    else:
        rating_lookup = combined_rankings_df.set_index("team")["combined_power_rating"]
        home_adv_lookup = combined_rankings_df.set_index("team")["home_adv"]
        rows = []
        for g in todays_games.itertuples(index=False):
            g = g._asdict()
            home, away = g["home_team"], g["away_team"]
            if home not in rating_lookup.index or away not in rating_lookup.index:
                continue
            home_r_, away_r_ = float(rating_lookup[home]), float(rating_lookup[away])
            diff = round(home_r_ - away_r_, 1)
            home_win_prob = round(
                100.0 / (1.0 + 10 ** ((away_r_ - (home_r_ + float(home_adv_lookup.get(home, 0.0)))) / 400.0)), 1
            )
            rows.append({
                "as_of_date": TODAY.isoformat(), "game_pk": g["game_pk"], "home_team": home,
                "home_combined_rating": home_r_, "away_team": away, "away_combined_rating": away_r_,
                "elo_diff": diff, "home_win_prob": home_win_prob,
            })
        daily_matchups_df = pd.DataFrame(rows, columns=matchup_cols)
    if not daily_matchups_df.empty:
        daily_matchups_df.to_csv(CSV_DIR / f"mlb_daily_matchups_{TODAY.isoformat()}.csv", index=False)

    # -------------------------------------------------------------
    # 12. Today's starting pitchers (NEW: not in the original notebook)
    #     Join today's probable starters against season pitcher ratings.
    # -------------------------------------------------------------
    starter_cols = ["as_of_date", "rank", "team", "pitcher", "rating", "opponent"]
    if todays_games.empty or not pitcher_ratings:
        daily_starters_df = pd.DataFrame(columns=starter_cols)
    else:
        rows = []
        for g in todays_games.itertuples(index=False):
            g = g._asdict()
            for team_key, pitcher_key, opp_key in (
                ("home_team", "home_probable_pitcher", "away_team"),
                ("away_team", "away_probable_pitcher", "home_team"),
            ):
                pitcher = g.get(pitcher_key)
                if pd.isna(pitcher):
                    continue
                rating = pitcher_ratings.get(pitcher)
                rows.append({
                    "as_of_date": TODAY.isoformat(), "team": g[team_key], "pitcher": pitcher,
                    "rating": round(rating, 1) if rating is not None else None,
                    "opponent": g[opp_key],
                })
        daily_starters_df = pd.DataFrame(rows)
        if not daily_starters_df.empty:
            daily_starters_df = daily_starters_df.sort_values("rating", ascending=False, na_position="last").reset_index(drop=True)
            daily_starters_df.insert(1, "rank", range(1, len(daily_starters_df) + 1))
            daily_starters_df = daily_starters_df[starter_cols]
    if not daily_starters_df.empty:
        daily_starters_df.to_csv(CSV_DIR / f"mlb_daily_starting_pitchers_{TODAY.isoformat()}.csv", index=False)

    # -------------------------------------------------------------
    # 13. Predictions log: freeze today's picks (once), backfill outcomes
    # -------------------------------------------------------------
    update_predictions_log(schedule_df, daily_matchups_df)

    # -------------------------------------------------------------
    # 14. Write docs/data/latest.json + teams.json + predictions_summary.json
    # -------------------------------------------------------------
    write_latest_json(daily_matchups_df, combined_rankings_df, daily_starters_df, {
        "base_K": base_K, "base_home_adv": base_HOME_ADV, "base_season_revert": base_SEASON_REVERT,
        "use_mov": bool(use_mov), "use_pitcher_adj": bool(use_pitcher_adj), "use_team_home_adv": bool(use_team_home_adv),
    })
    with open(DOCS_DATA_DIR / "teams.json", "w") as f:
        json.dump(teams_json_blob(), f, indent=2)
    write_predictions_summary()

    print("=== Done ===")


# =======================================================================
# Predictions tracking
# =======================================================================
LOG_COLUMNS = [
    "date", "game_pk", "home_team", "away_team",
    "combined_elo_pick", "home_win_prob", "home_prob_pick", "prob_bucket",
    "actual_winner", "combined_elo_correct", "home_prob_correct",
]

BUCKET_EDGES = list(range(0, 101, 5))  # 0,5,10,...,100
BUCKET_LABELS = [f"{lo}-{hi}%" for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])]


def _bucket_for(prob: float) -> str:
    idx = min(int(prob // 5), len(BUCKET_LABELS) - 1)
    return BUCKET_LABELS[idx]


def update_predictions_log(schedule_df: pd.DataFrame, daily_matchups_df: pd.DataFrame):
    if PREDICTIONS_LOG_PATH.exists():
        log_df = pd.read_csv(PREDICTIONS_LOG_PATH)
    else:
        log_df = pd.DataFrame(columns=LOG_COLUMNS)

    # Force object dtype on columns that start all-null (CSV round-tripping
    # otherwise infers float64 for an all-NaN column, which then raises when
    # we try to write a string/bool into it below).
    for col in ("actual_winner", "combined_elo_correct", "home_prob_correct"):
        if col not in log_df.columns:
            log_df[col] = pd.Series(dtype=object)
        log_df[col] = log_df[col].astype(object)


    # --- (a) freeze today's picks, but only for game_pks not already logged ---
    if not daily_matchups_df.empty:
        already_logged = set(log_df["game_pk"]) if not log_df.empty else set()
        new_rows = []
        for row in daily_matchups_df.itertuples(index=False):
            row = row._asdict()
            if row["game_pk"] in already_logged:
                continue
            home_r, away_r = row["home_combined_rating"], row["away_combined_rating"]
            combined_pick = row["home_team"] if home_r >= away_r else row["away_team"]
            prob = row["home_win_prob"]
            prob_pick = row["home_team"] if prob >= 50 else row["away_team"]
            new_rows.append({
                "date": row["as_of_date"], "game_pk": row["game_pk"],
                "home_team": row["home_team"], "away_team": row["away_team"],
                "combined_elo_pick": combined_pick, "home_win_prob": prob,
                "home_prob_pick": prob_pick, "prob_bucket": _bucket_for(prob),
                "actual_winner": None, "combined_elo_correct": None, "home_prob_correct": None,
            })
        if new_rows:
            log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"  Predictions log: froze {len(new_rows)} new game(s)")

    # --- (b) backfill outcomes for any previously-logged, unresolved games ---
    if not log_df.empty:
        unresolved_mask = log_df["actual_winner"].isna()
        if unresolved_mask.any():
            scores = schedule_df.dropna(subset=["home_score", "away_score"]).set_index("game_pk")
            filled = 0
            for idx in log_df[unresolved_mask].index:
                game_pk = log_df.at[idx, "game_pk"]
                if game_pk not in scores.index:
                    continue
                s = scores.loc[game_pk]
                if isinstance(s, pd.DataFrame):  # duplicate game_pk safety
                    s = s.iloc[0]
                home_score, away_score = s["home_score"], s["away_score"]
                if home_score == away_score:
                    continue  # shouldn't happen in MLB, but skip rather than mis-grade
                winner = log_df.at[idx, "home_team"] if home_score > away_score else log_df.at[idx, "away_team"]
                log_df.at[idx, "actual_winner"] = winner
                log_df.at[idx, "combined_elo_correct"] = bool(winner == log_df.at[idx, "combined_elo_pick"])
                log_df.at[idx, "home_prob_correct"] = bool(winner == log_df.at[idx, "home_prob_pick"])
                filled += 1
            if filled:
                print(f"  Predictions log: backfilled outcomes for {filled} game(s)")

    log_df.to_csv(PREDICTIONS_LOG_PATH, index=False)


def write_predictions_summary():
    if not PREDICTIONS_LOG_PATH.exists():
        summary = {"combined_elo_record": None, "home_prob_record": None, "calibration": [], "generated_at": None}
    else:
        log_df = pd.read_csv(PREDICTIONS_LOG_PATH)
        resolved = log_df.dropna(subset=["actual_winner"]).copy()

        def _record(correct_col):
            if resolved.empty:
                return {"wins": 0, "losses": 0, "pct": None}
            wins = int(resolved[correct_col].sum())
            losses = int((~resolved[correct_col].astype(bool)).sum())
            total = wins + losses
            return {"wins": wins, "losses": losses, "pct": round(100 * wins / total, 1) if total else None}

        calibration = []
        if not resolved.empty:
            resolved["home_won"] = (resolved["actual_winner"] == resolved["home_team"])
            for label, lo, hi in zip(BUCKET_LABELS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
                bucket_rows = resolved[resolved["prob_bucket"] == label]
                n = len(bucket_rows)
                calibration.append({
                    "bucket": label,
                    "range_low": lo,
                    "range_high": hi,
                    "n_games": n,
                    "avg_predicted_home_win_pct": round(bucket_rows["home_win_prob"].mean(), 1) if n else None,
                    "observed_home_win_pct": round(100 * bucket_rows["home_won"].mean(), 1) if n else None,
                })

        summary = {
            "combined_elo_record": _record("combined_elo_correct"),
            "home_prob_record": _record("home_prob_correct"),
            "calibration": calibration,
            "n_resolved": int(len(resolved)),
            "n_pending": int(log_df["actual_winner"].isna().sum()),
            "generated_at": TODAY.isoformat(),
        }

    with open(DOCS_DATA_DIR / "predictions_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# =======================================================================
# Frontend JSON bundle
# =======================================================================
def write_latest_json(daily_matchups_df, combined_rankings_df, daily_starters_df, model_flags):
    def _team_block(name):
        meta = team_meta(name)
        return {"name": name, "abbr": meta["abbr"], "primary": meta["primary"],
                "secondary": meta["secondary"], "logo": logo_url(name)}

    matchups = []
    for row in daily_matchups_df.itertuples(index=False):
        row = row._asdict()
        matchups.append({
            "game_pk": int(row["game_pk"]),
            "home": {**_team_block(row["home_team"]), "combined_rating": row["home_combined_rating"]},
            "away": {**_team_block(row["away_team"]), "combined_rating": row["away_combined_rating"]},
            "elo_diff": row["elo_diff"],
            "home_win_prob": row["home_win_prob"],
        })

    rankings = []
    for row in combined_rankings_df.itertuples(index=False):
        row = row._asdict()
        rankings.append({
            "team": row["team"], "abbr": team_meta(row["team"])["abbr"],
            "combined_rank": int(row["combined_rank"]), "combined_power_rating": row["combined_power_rating"],
            "elo_rank": int(row["elo_rank"]), "elo_rating": row["elo_rating"],
            "home_adv": row["home_adv"],
            "avg_rotation_pitcher_rating": None if pd.isna(row["avg_rotation_pitcher_rating"]) else row["avg_rotation_pitcher_rating"],
        })

    starters = []
    if not daily_starters_df.empty:
        for row in daily_starters_df.itertuples(index=False):
            row = row._asdict()
            starters.append({
                "rank": int(row["rank"]), "team": row["team"], "abbr": team_meta(row["team"])["abbr"],
                "pitcher": row["pitcher"], "rating": row["rating"], "opponent": row["opponent"],
            })

    bundle = {
        "as_of_date": TODAY.isoformat(),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "model_flags": model_flags,
        "matchups": matchups,
        "rankings": rankings,
        "starting_pitchers": starters,
    }
    with open(DOCS_DATA_DIR / "latest.json", "w") as f:
        json.dump(bundle, f, indent=2, default=str)


if __name__ == "__main__":
    main()
