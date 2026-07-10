// MLB Elo site — reads docs/data/latest.json and
// docs/data/predictions_summary.json (both written daily by
// scripts/update_data.py) and renders every section. No build step,
// no framework — this file is the whole frontend.

const DATA_DIR = "data";

async function fetchJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

function fmtNum(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------------------------------------------------------------
// Color utilities — readable text on any team color
// ---------------------------------------------------------------
function hexToRgb(hex) {
  const h = (hex || "").replace("#", "");
  if (h.length !== 6) return { r: 110, g: 110, b: 110 };
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function luminance({ r, g, b }) {
  const lin = (c) => {
    c /= 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrastRatio(l1, l2) {
  const [hi, lo] = l1 >= l2 ? [l1, l2] : [l2, l1];
  return (hi + 0.05) / (lo + 0.05);
}

// Brighten very dark team colors just enough that the segment is
// visible against the card background, then pick whichever text
// color (chalk / near-black wall green) has the higher contrast
// against the final background. Handles all 30 team palettes.
const CHALK = "#F4F1E4";
const WALL_DARK = "#0A140E";

function barColors(hex) {
  let { r, g, b } = hexToRgb(hex);
  while (luminance({ r, g, b }) < 0.05 && (r < 255 || g < 255 || b < 255)) {
    r = Math.min(255, r + 12);
    g = Math.min(255, g + 12);
    b = Math.min(255, b + 12);
  }
  const L = luminance({ r, g, b });
  const chalkL = luminance(hexToRgb(CHALK));
  const darkL = luminance(hexToRgb(WALL_DARK));
  return {
    bg: `rgb(${r}, ${g}, ${b})`,
    text: contrastRatio(L, chalkL) >= contrastRatio(L, darkL) ? CHALK : WALL_DARK,
  };
}

// ---------------------------------------------------------------
// Header
// ---------------------------------------------------------------
function renderHeader(latest) {
  const dateEl = document.getElementById("as-of-date");
  const d = new Date(latest.as_of_date + "T00:00:00");
  dateEl.textContent = d.toLocaleDateString(undefined, {
    weekday: "long", month: "long", day: "numeric", year: "numeric",
  });

  const flags = latest.model_flags || {};
  const on = [];
  if (flags.use_mov) on.push("MOV");
  if (flags.use_pitcher_adj) on.push("Pitcher adj");
  if (flags.use_team_home_adv) on.push("Team home adv");
  document.getElementById("model-flags").textContent =
    on.length ? `Active layers: ${on.join(" · ")}` : "Base model only";
}

// ---------------------------------------------------------------
// Matchups
// ---------------------------------------------------------------
function renderMatchups(latest) {
  const el = document.getElementById("matchups-list");
  const games = latest.matchups || [];

  if (!games.length) {
    el.innerHTML = `<p class="loading-msg">No games found for today — either it's an off day, or today's slate hasn't loaded yet.</p>`;
    return;
  }

  el.innerHTML = games.map((g, i) => matchupCardHTML(g, i)).join("");

  el.querySelectorAll(".matchup-card").forEach((card) => {
    const toggle = () => {
      const open = card.classList.toggle("open");
      card.setAttribute("aria-expanded", open);
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
  });
}

function matchupCardHTML(g, i) {
  const homeProb = g.home_win_prob;
  const awayProb = +(100 - homeProb).toFixed(1);
  // keep each segment readably wide even at extreme splits
  const homeW = Math.max(10, Math.min(90, homeProb));
  const awayW = 100 - homeW;

  // contrast-aware segment colors (see barColors above)
  const awayC = barColors(g.away.primary);
  const homeC = barColors(g.home.primary);

  const logo = (team) => team.logo
    ? `<img src="${team.logo}" alt="" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'team-abbr-fallback',textContent:'${escapeHtml(team.abbr)}'}))">`
    : `<span class="team-abbr-fallback">${escapeHtml(team.abbr)}</span>`;

  return `
    <article class="matchup-card" tabindex="0" role="button" aria-expanded="false"
             style="animation-delay:${Math.min(i, 12) * 45}ms">
      <div class="matchup-teams">
        <div class="matchup-team away">
          ${logo(g.away)}
          <span class="team-name">${escapeHtml(g.away.abbr)}</span>
        </div>
        <span class="matchup-at">@</span>
        <div class="matchup-team home">
          ${logo(g.home)}
          <span class="team-name">${escapeHtml(g.home.abbr)}</span>
        </div>
      </div>
      <div class="prob-bar" title="Home win probability: ${homeProb}%">
        <div class="seg away" style="width:${awayW}%; background:${awayC.bg}; color:${awayC.text}">${awayProb}%</div>
        <div class="seg home" style="width:${homeW}%; background:${homeC.bg}; color:${homeC.text}">${homeProb}%</div>
        <span class="split-mark" style="left:${awayW}%" aria-hidden="true"></span>
      </div>
      <div class="matchup-expand">
        <div class="matchup-expand-inner">
          <div class="drawer-pad">
            <div class="expand-row"><span>${escapeHtml(g.away.name)} combined rating</span><strong>${fmtNum(g.away.combined_rating)}</strong></div>
            <div class="expand-row"><span>${escapeHtml(g.home.name)} combined rating</span><strong>${fmtNum(g.home.combined_rating)}</strong></div>
            <div class="expand-row"><span>Elo diff (home &minus; away, incl. home field)</span><strong>${fmtNum(g.elo_diff)}</strong></div>
            <div class="expand-row"><span>Home win probability</span><strong>${fmtNum(g.home_win_prob)}%</strong></div>
          </div>
        </div>
      </div>
      <div class="expand-hint">
        <span class="hint-hover">hover for the numbers &mdash; click to pin</span>
        <span class="hint-tap">tap for the numbers</span>
      </div>
    </article>
  `;
}

// ---------------------------------------------------------------
// Rankings table (sortable)
// ---------------------------------------------------------------
let rankingsData = [];
let rankingsSort = { key: "combined_rank", dir: "asc" };

function renderRankings(latest) {
  rankingsData = (latest.rankings || []).map((r) => ({
    ...r,
    rank_delta: r.elo_rank - r.combined_rank, // positive = combined rating boosted them
  }));
  drawRankingsTable();

  document.querySelectorAll("#rankings-table th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (rankingsSort.key === key) {
        rankingsSort.dir = rankingsSort.dir === "asc" ? "desc" : "asc";
      } else {
        rankingsSort = { key, dir: "asc" };
      }
      drawRankingsTable();
    });
  });
}

function drawRankingsTable() {
  const { key, dir } = rankingsSort;
  const sorted = [...rankingsData].sort((a, b) => {
    let av = a[key], bv = b[key];
    if (typeof av === "string") { av = av.toLowerCase(); bv = bv.toLowerCase(); }
    if (av < bv) return dir === "asc" ? -1 : 1;
    if (av > bv) return dir === "asc" ? 1 : -1;
    return 0;
  });

  document.querySelectorAll("#rankings-table th.sortable").forEach((th) => {
    const isSorted = th.dataset.sort === key;
    th.classList.toggle("is-sorted", isSorted);
    if (isSorted) th.dataset.dir = dir; else th.removeAttribute("data-dir");
  });

  const body = document.getElementById("rankings-body");
  if (!sorted.length) {
    body.innerHTML = `<tr><td colspan="6" class="loading-msg">No ranking data available.</td></tr>`;
    return;
  }

  body.innerHTML = sorted.map((r) => {
    const delta = r.rank_delta;
    const deltaClass = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "delta-flat";
    const deltaSign = delta > 0 ? "+" : "";
    return `
      <tr>
        <td>${r.combined_rank}</td>
        <td>${escapeHtml(r.team)}</td>
        <td>${fmtNum(r.combined_power_rating)}</td>
        <td>${r.elo_rank}</td>
        <td>${fmtNum(r.elo_rating)}</td>
        <td class="${deltaClass}">${deltaSign}${delta}</td>
      </tr>
    `;
  }).join("");
}

// ---------------------------------------------------------------
// Starting pitchers
// ---------------------------------------------------------------
function renderPitchers(latest) {
  const body = document.getElementById("pitchers-body");
  const starters = latest.starting_pitchers || [];

  if (!starters.length) {
    body.innerHTML = `<tr><td colspan="5" class="loading-msg">No probable starters loaded for today (the pitcher-adjustment layer may be off, or lineups aren't posted yet).</td></tr>`;
    return;
  }

  body.innerHTML = starters.map((p) => `
    <tr>
      <td>${p.rank}</td>
      <td>${escapeHtml(p.pitcher)}</td>
      <td>${escapeHtml(p.abbr)}</td>
      <td>vs ${escapeHtml(p.opponent)}</td>
      <td>${fmtNum(p.rating)}</td>
    </tr>
  `).join("");
}

// ---------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------
function renderDownloads(latest) {
  const date = latest.as_of_date;
  const files = [
    { label: "Combined Rankings", name: `mlb_combined_rankings_${date}.csv` },
    { label: "Today's Matchups", name: `mlb_daily_matchups_${date}.csv` },
    { label: "Season Elo Rankings", name: `mlb_elo_rankings_${date}.csv` },
    { label: "Pitcher Ratings", name: `mlb_pitcher_ratings_${date}.csv` },
    { label: "Today's Starting Pitchers", name: `mlb_daily_starting_pitchers_${date}.csv` },
    { label: "Team Home Advantage", name: `mlb_team_home_adv_${date}.csv` },
    { label: "Full Predictions Log", name: `predictions_log.csv`, root: true },
  ];

  const el = document.getElementById("downloads-list");
  el.innerHTML = files.map((f) => {
    const href = f.root ? `${DATA_DIR}/${f.name}` : `${DATA_DIR}/csv/${f.name}`;
    return `<a class="download-btn" href="${href}" download><span class="icon">&#8595;</span>${escapeHtml(f.label)}</a>`;
  }).join("");
}

// ---------------------------------------------------------------
// Prediction trackers
// ---------------------------------------------------------------
function renderTrackers(summary) {
  fillRecordCard("record-combined", summary.combined_elo_record);
  fillRecordCard("record-homeprob", summary.home_prob_record);

  document.getElementById("n-resolved").textContent = summary.n_resolved ?? "0";
  document.getElementById("n-pending").textContent = summary.n_pending ?? "0";
  document.getElementById("tracking-since").textContent = summary.generated_at
    ? `updated ${summary.generated_at}` : "";

  const body = document.getElementById("calibration-body");
  const rows = (summary.calibration || []).filter((b) => b.n_games > 0);

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="5" class="loading-msg">No resolved games yet — the calibration table fills in as the season plays out.</td></tr>`;
    return;
  }

  body.innerHTML = rows.map((b) => {
    const predicted = b.avg_predicted_home_win_pct;
    const observed = b.observed_home_win_pct;
    const diff = observed - predicted;
    const diffClass = Math.abs(diff) <= 5 ? "delta-flat" : diff > 0 ? "delta-up" : "delta-down";
    const diffLabel = `${diff > 0 ? "+" : ""}${fmtNum(diff)} pts`;

    return `
      <tr>
        <td>${b.range_low}&ndash;${b.range_high}%</td>
        <td>${b.n_games}</td>
        <td>${fmtNum(predicted)}%</td>
        <td>${fmtNum(observed)}%</td>
        <td>
          <span class="calibration-bar">
            <span class="fill" style="width:${Math.min(100, observed)}%"></span>
            <span class="marker" style="left:${Math.min(100, predicted)}%"></span>
          </span>
          <span class="${diffClass}">${diffLabel}</span>
        </td>
      </tr>
    `;
  }).join("");
}

function fillRecordCard(id, record) {
  const card = document.getElementById(id);
  if (!record || record.pct === null) {
    card.querySelector(".w").textContent = "0";
    card.querySelector(".l").textContent = "0";
    card.querySelector(".record-pct").textContent = "no resolved games yet";
    return;
  }
  card.querySelector(".w").textContent = record.wins;
  card.querySelector(".l").textContent = record.losses;
  card.querySelector(".record-pct").textContent = `${fmtNum(record.pct)}% correct`;
}

// ---------------------------------------------------------------
// Boot
// ---------------------------------------------------------------
async function init() {
  try {
    const latest = await fetchJSON(`${DATA_DIR}/latest.json`);
    renderHeader(latest);
    renderMatchups(latest);
    renderRankings(latest);
    renderPitchers(latest);
    renderDownloads(latest);
  } catch (err) {
    console.error(err);
    document.getElementById("matchups-list").innerHTML =
      `<p class="loading-msg">Couldn't load today's data yet. Check back after the next scheduled update.</p>`;
  }

  try {
    const summary = await fetchJSON(`${DATA_DIR}/predictions_summary.json`);
    renderTrackers(summary);
  } catch (err) {
    console.error(err);
    document.getElementById("calibration-body").innerHTML =
      `<tr><td colspan="5" class="loading-msg">Couldn't load the prediction tracker yet.</td></tr>`;
  }
}

init();
