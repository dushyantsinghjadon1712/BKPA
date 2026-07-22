const API_BASE = ""; // same origin — FastAPI serves this file, so relative paths work

// --- tab switching ---
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

function fillSelect(select, values, placeholder) {
  select.innerHTML = "";
  if (placeholder) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = placeholder;
    select.appendChild(opt);
  }
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    select.appendChild(opt);
  });
}

function renderTable(container, records) {
  if (!records || records.length === 0) {
    container.innerHTML = '<p class="note">No results.</p>';
    return;
  }
  const cols = Object.keys(records[0]);
  let html = "<table><thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  records.forEach(r => {
    html += "<tr>" + cols.map(c => `<td>${r[c] === null || r[c] === undefined ? "-" : r[c]}</td>`).join("") + "</tr>";
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// --- init: populate dropdowns from /api/meta ---
async function init() {
  try {
    const meta = await apiGet("/api/meta");

    fillSelect(document.getElementById("bm-crop"), meta.crops);
    fillSelect(document.getElementById("tr-crop"), meta.crops);
    fillSelect(document.getElementById("fc-crop"), meta.crops);
    fillSelect(document.getElementById("ny-state"), meta.national_states);

    document.getElementById("model-meta").textContent =
      `National yield model — MAE ${meta.yield_model_metrics.MAE}, R² ${meta.yield_model_metrics.R2}`;

    // when crop changes on trend/forecast tabs, refresh market dropdown
    document.getElementById("tr-crop").addEventListener("change", () => refreshMarkets("tr"));
    document.getElementById("fc-crop").addEventListener("change", () => refreshMarkets("fc"));
    refreshMarkets("tr");
    refreshMarkets("fc");
  } catch (e) {
    document.body.insertAdjacentHTML("afterbegin", `<p class="error">Failed to load API: ${e.message}. Is the backend running?</p>`);
  }
}

async function refreshMarkets(prefix) {
  const crop = document.getElementById(`${prefix}-crop`).value;
  if (!crop) return;
  const bm = await apiGet(`/api/best-mandi?crop=${encodeURIComponent(crop)}&top_n=20`);
  const markets = bm.map(r => r.Market);
  const select = document.getElementById(`${prefix}-market`);
  const placeholder = prefix === "tr" ? "All markets" : null;
  fillSelect(select, markets, placeholder);
}

// --- tab actions ---
async function loadBestMandi() {
  const crop = document.getElementById("bm-crop").value;
  const topN = document.getElementById("bm-topn").value;
  const container = document.getElementById("bm-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    const data = await apiGet(`/api/best-mandi?crop=${encodeURIComponent(crop)}&top_n=${topN}`);
    renderTable(container, data);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

async function loadTrend() {
  const crop = document.getElementById("tr-crop").value;
  const market = document.getElementById("tr-market").value;
  const container = document.getElementById("tr-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    let url = `/api/price-trend?crop=${encodeURIComponent(crop)}`;
    if (market) url += `&market=${encodeURIComponent(market)}`;
    const data = await apiGet(url);
    renderTable(container, data);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

async function loadVolatility() {
  const container = document.getElementById("vol-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    const data = await apiGet("/api/volatility");
    renderTable(container, data);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

async function loadRecommendation() {
  const season = document.getElementById("rec-season").value;
  const topN = document.getElementById("rec-topn").value;
  const container = document.getElementById("rec-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    const data = await apiGet(`/api/crop-recommendation?season=${encodeURIComponent(season)}&top_n=${topN}`);
    renderTable(container, data);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

async function loadForecast() {
  const crop = document.getElementById("fc-crop").value;
  const market = document.getElementById("fc-market").value;
  const periods = document.getElementById("fc-periods").value;
  const container = document.getElementById("fc-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    const data = await apiGet(`/api/forecast?crop=${encodeURIComponent(crop)}&market=${encodeURIComponent(market)}&periods=${periods}`);
    const metricsHtml = `
      <div class="metric-row">
        <div class="metric-box">MAE<b>₹${data.metrics.MAE}</b></div>
        <div class="metric-box">MAPE<b>${data.metrics.MAPE_pct}%</b></div>
      </div>`;
    container.innerHTML = metricsHtml;
    const tableDiv = document.createElement("div");
    container.appendChild(tableDiv);
    renderTable(tableDiv, data.forecast);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

async function loadNational() {
  const state = document.getElementById("ny-state").value;
  const season = document.getElementById("ny-season").value;
  const topN = document.getElementById("ny-topn").value;
  const container = document.getElementById("ny-result");
  container.innerHTML = '<p class="loading">Loading...</p>';
  try {
    const data = await apiGet(`/api/national-yield?state=${encodeURIComponent(state)}&season=${encodeURIComponent(season)}&top_n=${topN}`);
    renderTable(container, data);
  } catch (e) {
    container.innerHTML = `<p class="error">${e.message}</p>`;
  }
}

init();
