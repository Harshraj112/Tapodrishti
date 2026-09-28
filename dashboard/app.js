const state = {
  overview: null,
  selectedWellId: null,
  well: null,
  sortDescending: true,
  toastTimer: null,
};
const DASHBOARD_POLL_MS = 60_000;

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));
const number = (value, digits = 1) => Number.isFinite(Number(value))
  ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })
  : '--';
const titleCase = (value) => String(value || 'unknown').replaceAll('_', ' ')
  .replace(/\b\w/g, (letter) => letter.toUpperCase());
const localTime = (value) => value ? new Date(value).toLocaleString() : 'not available';

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function refreshIcons() {
  if (window.lucide?.createIcons) window.lucide.createIcons();
}

function notify(message, isError = false) {
  const toast = byId('toast');
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.classList.add('show');
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove('show'), 3200);
}

function riskTone(tier) {
  const normalized = String(tier || 'low').toLowerCase();
  return ['high', 'medium', 'low'].includes(normalized) ? normalized : 'low';
}

function actionClass(action) {
  return String(action || 'hold').toLowerCase().replace(/[^a-z_]/g, '');
}

function renderSummary(overview) {
  const summary = overview.summary;
  byId('metric-wells').textContent = summary.well_count;
  byId('metric-sor').textContent = number(summary.median_sor, 2);
  byId('metric-risk').textContent = summary.high_risk_count;
  byId('metric-fillage').textContent = number(summary.mean_fillage_pct, 1);
  byId('fleet-count').textContent = overview.fleet.length;
  byId('updated-at').textContent = `Snapshot ${localTime(overview.refreshed_at)}`;
  byId('model-training-label').textContent = `Saved model ${localTime(overview.model_trained_at)}`;
  byId('data-as-of').textContent = `Data through ${localTime(overview.data_as_of)}`;
  byId('next-refresh-at').textContent = `Next refresh ${localTime(overview.next_refresh_at)}`;
  byId('well-count-label').textContent = `${summary.well_count}-well production fleet`;
  byId('data-source-label').textContent = `${overview.model_artifact} / FIXED WEIGHTS`;
  byId('sidebar-status').textContent = overview.last_refresh_error
    ? 'Last refresh failed; serving previous snapshot'
    : (overview.data_matches_model_training ? 'Saved model data matched' : 'Current data differs from model training data');
  byId('footer-source').textContent = `${overview.data_source} · source date ${localTime(overview.data_as_of)} · decision support only`;
  renderAlerts(overview.alerts || []);
}

function renderAlerts(alerts) {
  const panel = byId('alert-panel');
  panel.hidden = alerts.length === 0;
  byId('alert-count').textContent = alerts.length;
  byId('alert-list').innerHTML = alerts.slice(-8).reverse().map((alert) => {
    const severity = ['high', 'medium', 'info'].includes(alert.severity) ? alert.severity : 'info';
    return `<button class="alert-item alert-${severity}" type="button" data-well="${escapeHtml(alert.well_id)}">
      <span class="alert-symbol"><i data-lucide="${severity === 'high' ? 'triangle-alert' : 'arrow-right-left'}"></i></span>
      <span class="alert-text"><strong>${escapeHtml(alert.well_id)} · ${escapeHtml(titleCase(alert.type))}</strong><span>${escapeHtml(alert.message)}</span></span>
      <time>${escapeHtml(localTime(alert.detected_at))}</time>
    </button>`;
  }).join('');
  byId('alert-list').querySelectorAll('[data-well]').forEach((button) => {
    button.addEventListener('click', () => selectWell(button.dataset.well));
  });
  refreshIcons();
}

function visibleFleet() {
  const query = byId('well-search').value.trim().toLowerCase();
  const filter = byId('well-filter').value;
  const fleet = [...(state.overview?.fleet || [])];
  const filtered = fleet.filter((well) => {
    const matchesQuery = !query || `${well.well_id} ${well.card_label} ${well.risk_tier} ${well.srp_action}`
      .toLowerCase().includes(query);
    const matchesFilter = filter === 'all'
      || (filter === 'attention' && (well.risk_tier !== 'low' || well.srp_action !== 'hold'))
      || (filter === 'rod_float' && well.card_label === 'rod_float')
      || (filter === 'high_risk' && well.risk_tier === 'high');
    return matchesQuery && matchesFilter;
  });
  return filtered.sort((left, right) => (left.failure_probability_30d - right.failure_probability_30d)
    * (state.sortDescending ? -1 : 1));
}

function renderFleet() {
  const fleet = visibleFleet();
  byId('fleet-footnote').textContent = `${fleet.length} shown · Select a row to inspect`;
  const body = byId('fleet-body');
  if (!fleet.length) {
    body.innerHTML = '<tr><td class="loading-row" colspan="6">No wells match this filter.</td></tr>';
    return;
  }
  body.innerHTML = fleet.map((well) => {
    const fillage = Math.max(0, Math.min(100, Number(well.latest_fillage_pct) || 0));
    const tier = riskTone(well.risk_tier);
    const cardClass = actionClass(well.card_label);
    const action = actionClass(well.srp_action);
    return `<tr data-well="${escapeHtml(well.well_id)}" tabindex="0" role="button" aria-label="Open ${escapeHtml(well.well_id)} details" class="${well.well_id === state.selectedWellId ? 'selected' : ''}">
      <td><span class="well-id">${escapeHtml(well.well_id)}</span></td>
      <td><span class="cell-number">${number(well.latest_sor, 2)}</span></td>
      <td><div class="fillage-wrap"><span class="cell-number">${number(fillage, 0)}%</span><span class="mini-meter"><i style="width:${fillage}%"></i></span></div></td>
      <td><span class="pill pill-${cardClass}">${escapeHtml(titleCase(well.card_label))}</span></td>
      <td><span class="pill pill-${tier}">${escapeHtml(titleCase(tier))} · ${number((Number(well.failure_probability_30d) || 0) * 100, 1)}%</span></td>
      <td><span class="pill pill-${action}">${escapeHtml(titleCase(well.srp_action))}</span></td>
    </tr>`;
  }).join('');
  body.querySelectorAll('tr[data-well]').forEach((row) => {
    row.addEventListener('click', () => selectWell(row.dataset.well));
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectWell(row.dataset.well);
      }
    });
  });
}

function renderPriorityList() {
  const sorted = [...(state.overview?.fleet || [])]
    .sort((left, right) => right.failure_probability_30d - left.failure_probability_30d)
    .slice(0, 5);
  byId('priority-list').innerHTML = sorted.map((well, index) => {
    const tier = riskTone(well.risk_tier);
    return `<button class="priority-item risk-${tier}" type="button" data-well="${escapeHtml(well.well_id)}">
      <span class="priority-rank">${String(index + 1).padStart(2, '0')}</span>
      <span class="priority-info"><strong>${escapeHtml(well.well_id)}</strong><span>${escapeHtml(titleCase(well.card_label))} · ${escapeHtml(titleCase(well.srp_action))}</span></span>
      <span class="priority-prob">${number((Number(well.failure_probability_30d) || 0) * 100, 1)}%</span>
    </button>`;
  }).join('');
  byId('priority-list').querySelectorAll('[data-well]').forEach((button) => {
    button.addEventListener('click', () => selectWell(button.dataset.well));
  });
}

function buildLineChart(containerId, rows, key, color, fixedRange = null) {
  const container = byId(containerId);
  const values = rows.map((row) => Number(row[key])).filter(Number.isFinite);
  if (values.length < 2) {
    container.innerHTML = '<div class="chart-empty">Not enough readings to plot</div>';
    return;
  }
  const width = 620;
  const height = 132;
  const left = 35;
  const right = 610;
  const top = 12;
  const bottom = 111;
  const min = fixedRange ? fixedRange[0] : Math.min(...values);
  const max = fixedRange ? fixedRange[1] : Math.max(...values);
  const spread = Math.max(max - min, Math.abs(max) * 0.08, 1);
  const floor = fixedRange ? min : min - spread * 0.12;
  const ceiling = fixedRange ? max : max + spread * 0.12;
  const points = values.map((value, index) => {
    const x = left + (right - left) * index / (values.length - 1);
    const y = bottom - ((value - floor) / (ceiling - floor || 1)) * (bottom - top);
    return [x, Math.max(top, Math.min(bottom, y))];
  });
  const line = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area = `${line} L${right},${bottom} L${left},${bottom} Z`;
  const grid = [0, 1, 2].map((index) => {
    const y = top + (bottom - top) * index / 2;
    return `<line class="chart-grid-line" x1="${left}" x2="${right}" y1="${y}" y2="${y}"/>`;
  }).join('');
  const last = points.at(-1);
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(key.replaceAll('_', ' '))} across the latest production cycle" preserveAspectRatio="none">
    ${grid}<path class="chart-area-path" d="${area}" fill="${color}"/>
    <path class="chart-line" d="${line}" stroke="${color}"/>
    <circle class="chart-point" cx="${last[0]}" cy="${last[1]}" r="4" fill="${color}"/>
    <text x="0" y="15" fill="#9aa49d" font-size="9" font-family="DM Mono, monospace">${escapeHtml(number(max, 0))}</text>
    <text x="0" y="111" fill="#9aa49d" font-size="9" font-family="DM Mono, monospace">${escapeHtml(number(min, 0))}</text>
  </svg>`;
}

function renderCycleHistory(cycles) {
  const recent = cycles.slice(-8);
  const maxOil = Math.max(...recent.map((cycle) => Number(cycle.cycle_cum_oil_bbl) || 0), 1);
  byId('cycle-history').innerHTML = recent.map((cycle) => {
    const oil = Number(cycle.cycle_cum_oil_bbl) || 0;
    const height = Math.max(5, oil / maxOil * 52);
    return `<div class="cycle-column" title="Cycle ${cycle.cycle_number}: ${number(oil, 0)} bbl oil">
      <span class="cycle-bar-track"><i class="cycle-bar" style="height:${height}px"></i></span>
      <strong>${number(cycle.cycle_sor, 2)}</strong><span>C${cycle.cycle_number}</span>
    </div>`;
  }).join('');
}

function renderWell(detail) {
  state.well = detail;
  const master = detail.well_master;
  const cycles = detail.cycle_history || [];
  const daily = detail.latest_cycle_daily || [];
  const latest = cycles.at(-1) || {};
  byId('detail-well-title').textContent = detail.well_id;
  byId('well-summary-values').innerHTML = `
    <div class="well-fact"><span>API GRAVITY</span><strong>${number(master.api_gravity, 2)} deg</strong></div>
    <div class="well-fact"><span>RESERVOIR TEMP</span><strong>${number(master.reservoir_temp_c, 1)} C</strong></div>
    <div class="well-fact"><span>LATEST CYCLE</span><strong>C${detail.latest_cycle}</strong></div>
    <div class="well-fact"><span>LATEST OIL</span><strong>${number(latest.cycle_cum_oil_bbl, 0)} bbl</strong></div>`;
  buildLineChart('oil-chart', daily, 'oil_rate_bopd', '#338364');
  buildLineChart('viscosity-chart', daily, 'viscosity_cp', '#c07146');
  buildLineChart('fillage-chart', daily, 'pump_fillage_pct', '#6386a0', [0, 100]);
  renderCycleHistory(cycles);
  renderFaultAndRisk(detail);
  renderAdvice(detail);
  byId('optimizer-results').hidden = true;
  byId('forecast-results').hidden = true;
  refreshIcons();
}

function renderFaultAndRisk(detail) {
  const classification = detail.card_classification;
  const latestCard = (detail.sample_cards || []).at(-1);
  byId('fault-result').innerHTML = classification
    ? `<div><div class="fault-name">${escapeHtml(titleCase(classification.predicted_label))}</div><div class="fault-context"><span class="status-dot"></span>${latestCard ? `Latest card sample · day ${latestCard.day_index}` : 'Card model prediction'}</div></div><span class="fault-confidence">${number(Number(classification.confidence) * 100, 0)}% conf.</span>`
    : '<div class="fault-name">No card sample</div>';
  const risk = detail.current_rod_risk || {};
  const probability = Math.max(0, Math.min(1, Number(risk.failure_probability_30d) || 0));
  const tier = riskTone(risk.risk_tier);
  byId('risk-probability').textContent = `${number(probability * 100, 1)}%`;
  byId('risk-tier').innerHTML = `<strong>${escapeHtml(tier)} risk</strong>`;
  byId('risk-meter-fill').style.width = `${probability * 100}%`;
  byId('risk-meter-fill').className = tier;
}

function renderAdvice(detail) {
  const advice = detail.current_advice || {};
  byId('advice-action').textContent = titleCase(advice.action || 'hold');
  byId('advice-rationale').textContent = advice.rationale || 'No advisory available.';
  byId('advice-setpoints').innerHTML = `
    <span class="setpoint">SPM ${number(advice.recommended_spm, 2)}</span>
    <span class="setpoint">Stroke ${number(advice.recommended_stroke_length_in, 0)} in</span>
    <span class="setpoint">Viscosity ${escapeHtml(detail.viscosity_trend || 'stable')}</span>`;
}

async function selectWell(wellId) {
  if (!wellId) return;
  state.selectedWellId = wellId;
  byId('selected-well').value = wellId;
  renderFleet();
  byId('detail-well-title').textContent = `${wellId} / loading`;
  try {
    renderWell(await api(`/api/wells/${encodeURIComponent(wellId)}`));
  } catch (error) {
    notify(error.message, true);
    byId('detail-well-title').textContent = 'Well data unavailable';
  }
}

function renderOptimizer(result) {
  const params = result.recommended_params;
  const forecast = result.predicted_forecast;
  const economics = result.economics_breakdown;
  byId('optimizer-results').hidden = false;
  byId('optimizer-state').textContent = `${state.selectedWellId} / candidate cycle`;
  byId('optimizer-values').innerHTML = `<div class="optimizer-grid">
    <div class="optimizer-value"><span>STEAM</span><strong>${number(params.steam_volume_cwe_bbl, 0)} bbl</strong></div>
    <div class="optimizer-value"><span>INJECTION PRESSURE</span><strong>${number(params.injection_pressure_kpa, 0)} kPa</strong></div>
    <div class="optimizer-value"><span>SOAK</span><strong>${number(params.soak_time_days, 1)} days</strong></div>
    <div class="optimizer-value"><span>FORECAST OIL</span><strong>${number(forecast.predicted_cycle_oil_bbl, 0)} bbl</strong></div>
    <div class="optimizer-value"><span>FORECAST SOR</span><strong>${number(forecast.predicted_cycle_sor, 2)}</strong></div>
    <div class="optimizer-value"><span>ROD-FLOAT DAYS</span><strong>${number(economics.predicted_rod_float_days, 0)} days</strong></div>
  </div><div class="optimizer-net">Net model value INR ${number(result.predicted_economic_value_inr, 0)}</div>`;
  byId('optimizer-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function runCssScenario() {
  const button = byId('optimize-button');
  if (!state.selectedWellId) return;
  button.disabled = true;
  button.querySelector('span').textContent = 'Searching candidates...';
  byId('optimizer-results').hidden = false;
  byId('optimizer-state').textContent = 'Gaussian-process search in progress';
  byId('optimizer-values').innerHTML = '';
  try {
    renderOptimizer(await api(`/api/wells/${encodeURIComponent(state.selectedWellId)}/css-recommendation`, { method: 'POST' }));
  } catch (error) {
    byId('optimizer-state').textContent = 'Scenario unavailable';
    notify(error.message, true);
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run CSS scenario';
  }
}

function renderCssForecast(result) {
  const forecast = result.forecast;
  byId('forecast-results').hidden = false;
  byId('forecast-values').innerHTML = `
    <div class="forecast-value"><span>POST-SOAK TEMP</span><strong>${number(forecast.peak_post_soak_temp_c, 1)} C</strong></div>
    <div class="forecast-value"><span>THERMAL DECAY</span><strong>${number(forecast.thermal_decay_rate_per_day, 4)} /day</strong></div>
    <div class="forecast-value"><span>CYCLE OIL</span><strong>${number(forecast.predicted_cycle_oil_bbl, 0)} bbl</strong></div>
    <div class="forecast-value"><span>PREDICTED SOR</span><strong>${number(forecast.predicted_cycle_sor, 2)}</strong></div>
    <div class="forecast-value"><span>PRODUCTION</span><strong>${number(forecast.predicted_production_days, 0)} days</strong></div>`;
  const trainingState = result.model.training_data_matches_current
    ? 'Current dataset fingerprint matches the training data.'
    : 'Current dataset differs from the artifact training snapshot.';
  byId('forecast-model-stamp').textContent = `${result.model.artifact} · trained ${localTime(result.model.trained_at)} · ${trainingState}`;
}

async function runDirectForecast(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button[type="submit"]');
  const values = Object.fromEntries(new FormData(form).entries());
  for (const key of Object.keys(values)) values[key] = Number(values[key]);
  button.disabled = true;
  button.querySelector('span').textContent = 'Predicting...';
  byId('forecast-results').hidden = false;
  byId('forecast-values').textContent = 'Running saved model...';
  try {
    const result = await api(`/api/wells/${encodeURIComponent(state.selectedWellId)}/css-forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values),
    });
    renderCssForecast(result);
  } catch (error) {
    byId('forecast-values').textContent = '';
    byId('forecast-model-stamp').textContent = error.message;
    notify(error.message, true);
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Predict this design';
  }
}

async function loadDashboard(silent = false) {
  if (!silent) byId('fleet-body').innerHTML = '<tr><td class="loading-row" colspan="6">Loading fleet signals...</td></tr>';
  try {
    state.overview = await api('/api/overview');
    renderSummary(state.overview);
    renderPriorityList();
    const select = byId('selected-well');
    select.innerHTML = state.overview.fleet.map((well) => `<option value="${escapeHtml(well.well_id)}">${escapeHtml(well.well_id)}</option>`).join('');
    renderFleet();
    const selected = state.selectedWellId || [...state.overview.fleet]
      .sort((left, right) => right.failure_probability_30d - left.failure_probability_30d)[0]?.well_id;
    if (selected) await selectWell(selected);
    refreshIcons();
  } catch (error) {
    byId('sidebar-status').textContent = 'Model service unavailable';
    byId('fleet-body').innerHTML = '<tr><td class="loading-row" colspan="6">Unable to load model data. Check the backend process.</td></tr>';
    notify(error.message, true);
  }
}

async function reloadModels() {
  const button = byId('refresh-button');
  button.disabled = true;
  button.classList.add('refreshing');
  try {
    const result = await api('/api/refresh', { method: 'POST' });
    if (result.refreshing) {
      notify('A data and model refresh is already running.');
      return;
    }
    await loadDashboard(true);
    notify(result.refreshed ? 'Source data and saved model artifact reloaded.' : result.error, !result.refreshed);
  } catch (error) {
    notify(error.message, true);
  } finally {
    button.disabled = false;
    button.classList.remove('refreshing');
  }
}

byId('well-search').addEventListener('input', renderFleet);
byId('well-filter').addEventListener('change', renderFleet);
byId('selected-well').addEventListener('change', (event) => selectWell(event.target.value));
byId('optimize-button').addEventListener('click', runCssScenario);
byId('css-forecast-form').addEventListener('submit', runDirectForecast);
byId('refresh-button').addEventListener('click', reloadModels);
byId('sort-risk-button').addEventListener('click', () => {
  state.sortDescending = !state.sortDescending;
  byId('sort-risk-button').querySelector('span').textContent = state.sortDescending ? 'Risk priority' : 'Lowest risk';
  renderFleet();
});
document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', () => {
  document.querySelectorAll('.nav-link').forEach((item) => item.classList.remove('active'));
  link.classList.add('active');
}));

loadDashboard();
window.setInterval(() => {
  if (!document.hidden) loadDashboard(true);
}, DASHBOARD_POLL_MS);