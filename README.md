# MLB Elo

A daily-updating MLB power-rating site: an auto-tuned Elo model (win/loss +
optional margin-of-victory + optional starting-pitcher adjustment + optional
team/park home-field advantage) predicts every game on today's slate, tracks
its own accuracy over the season, and publishes everything as a static site
on GitHub Pages.

**Live data flow:** a GitHub Actions cron job runs the model twice a day,
writes fresh CSV/JSON snapshots into the repo, and commits them. The static
frontend in `docs/` reads those files — there's no backend server, no
database, nothing else to host.

## Repo layout

```
scripts/
  update_data.py       # the whole pipeline: pull schedule -> tune model ->
                        # build ratings -> write CSVs/JSON -> update the
                        # predictions log
  teams.py              # team name -> logo/color lookup
data/
  cache/                # cached season schedules (committed — this is what
                        # makes re-runs cheap; a completed season is never
                        # re-fetched)
  raw/elo/              # full game-by-game elo history
docs/                   # <- GitHub Pages serves from here
  index.html, styles.css, app.js
  data/
    latest.json          # today's matchups + rankings + starters, one fetch
    teams.json           # team colors/logos lookup
    predictions_log.csv  # append-only, every pick ever made + outcome
    predictions_summary.json  # pre-aggregated tracker numbers
    csv/                 # dated CSV snapshots, for the download buttons
.github/workflows/
  daily-update.yml       # the cron job
```

## One-time setup

1. **Push this repo to GitHub.**

2. **Let Actions write to the repo.** Settings → Actions → General →
   Workflow permissions → select **"Read and write permissions"** → Save.
   (The workflow file also declares `permissions: contents: write`, but the
   repo-level setting has to allow it too.)

3. **Turn on GitHub Pages.** Settings → Pages → Source: **Deploy from a
   branch** → Branch: `main`, folder: **`/docs`** → Save.

4. **Run the workflow once manually** so there's data before anyone visits:
   Actions tab → "Update MLB Elo data" → **Run workflow**. First run pulls
   6 seasons of schedule history and fits the model from scratch, so expect
   it to take a few minutes — after that, `data/cache/` is committed and
   every later run only re-fetches a trailing few days.

That's it — the site will be live at
`https://<your-username>.github.io/<repo-name>/`, and refreshes automatically
at ~10am and ~4pm ET (edit the two `cron:` lines in
`.github/workflows/daily-update.yml` to change the schedule).

## Running locally

```bash
pip install -r requirements.txt
python scripts/update_data.py
python -m http.server 8000 --directory docs   # then open localhost:8000
```

## How the prediction trackers work

Every time a game first appears in `docs/data/predictions_log.csv`, its pick
is frozen at that moment — a later same-day run (e.g. after a pitcher
change) never rewrites an existing row, so the record reflects what was
actually predicted, not a moving target. Each run also scans unresolved rows
and fills in the final score once a game finishes. Three views come off that
one log:

- **Combined-elo pick record** — win/loss for always taking the higher
  `combined_power_rating` team, home field ignored.
- **Home win% pick record** — win/loss for taking whichever team the
  home-field-adjusted probability favors.
- **Calibration table** — games bucketed by predicted home win % (5-point
  buckets, full 0–100% range), each bucket showing predicted vs. observed
  home win rate. In a well-calibrated model these should track closely in
  every row.

## Known limitations (inherited from the original model)

- The starting-pitcher layer scores a pitcher against "runs allowed while
  they started," which bundles in bullpen and fielding — a real limitation,
  not an oversight. A cleaner version would need per-pitcher innings/earned
  runs from box scores.
- `combined_power_rating` folds the rotation into team strength but keeps
  home-field advantage as a separate column, since home field only applies
  when a team is actually at home — summing it in would misrepresent road
  strength.
