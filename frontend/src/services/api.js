
import { postJSON, getDatasetId } from "../api/http";

const DEFAULT_API_BASE =
  typeof location !== 'undefined'
    ? `${location.protocol}//${location.hostname}:8000`
    : 'http://localhost:8000';

export const API_BASE =
  (window.API_BASE) || (location.protocol + '//' + location.hostname + ':8000');

// ------------------------- Legacy endpoints -------------------------
export async function postColumns(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/columns`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


export async function postAnalisar(fd) {
  const res = await fetch(`${API_BASE}/analisar`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postOptimize(payload = {}) {
  const datasetId = payload.dataset_id || getDatasetId();
  const targetName = payload.target_name ?? payload.target;
  const kMin = payload.k_min ?? payload.n_components_min ?? 1;
  const kMax = payload.k_max ?? payload.n_components_max;
  const resolvedMode =
    payload.mode ??
    (typeof payload.classification === "boolean"
      ? payload.classification
        ? "classification"
        : "regression"
      : undefined);

  const body = {
    ...payload,
    dataset_id: datasetId,
    target_name: targetName,
    mode: resolvedMode,
    validation_method: payload.validation_method,
    n_splits: payload.n_splits,
    repeats: payload.repeats,
    threshold: payload.threshold,
    n_components_min: kMin,
    n_components_max: kMax,
    metric_goal: payload.metric_goal,
    min_score: payload.min_score,
    spectral_range: payload.spectral_range,
    preprocess_grid: payload.preprocess_grid,
    sg_params: payload.sg_params,
    use_ipls: payload.use_ipls,
    ipls_max_intervals: payload.ipls_max_intervals,
    ipls_interval_width: payload.ipls_interval_width,
  };

  delete body.target;
  delete body.classification;

  const cleaned = Object.fromEntries(
    Object.entries(body).filter(([, value]) => value !== null && value !== undefined)
  );

  return postJSON(`${API_BASE}/optimize`, cleaned);
}

export async function getOptimizeStatus({ signal } = {}) {
  const res = await fetch(`${API_BASE}/optimize/status`, { signal });
  if (!res.ok) {
    let msg;
    try { const j = await res.json(); msg = j.detail || JSON.stringify(j); }
    catch { msg = await res.text(); }
    throw new Error(msg || 'Erro ao consultar status.');
  }
  return res.json();
}

export async function postReport(payload) {
  const res = await fetch(`${API_BASE}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function downloadReport(path) {
  const res = await fetch(`${API_BASE}/report/download?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

// --------------------- Helpers & normalisation ----------------------
function toNumberOrNull(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  const s = String(v).trim().replace(',', '.');
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

export function normalizeMatrix(rows) {
  return rows.map((r) => r.map(toNumberOrNull));
}

// ----------- New JSON endpoints with automatic normalisation ---------
export async function preprocess(X, y = null) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ X: normalizeMatrix(X), y }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


export async function train(X, y, nComponents = 10) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      X: normalizeMatrix(X),
      y,
      n_components: nComponents,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


export async function predict(X) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ X: normalizeMatrix(X) }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ------------- Wrappers that accept pre‑normalised payloads -----------
export async function postPreprocess(payload) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


function normalizeSpectralRange(value) {
  if (!value) return null;

  const toNumber = (v) => {
    if (typeof v === 'number') return Number.isFinite(v) ? v : null;
    if (v === null || v === undefined) return null;
    const s = String(v).trim().replace(',', '.');
    if (s === '') return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  };

  if (Array.isArray(value)) {
    for (const entry of value) {
      if (Array.isArray(entry) && entry.length >= 2) {
        const min = toNumber(entry[0]);
        const max = toNumber(entry[1]);
        if (min !== null && max !== null && min !== max) {
          return { min: Math.min(min, max), max: Math.max(min, max) };
        }
      }
    }
    return null;
  }

  if (typeof value === 'object') {
    const min = toNumber(value.min ?? value[0]);
    const max = toNumber(value.max ?? value[1]);
    if (min !== null && max !== null && min !== max) {
      return { min: Math.min(min, max), max: Math.max(min, max) };
    }
    return null;
  }

  if (typeof value === 'string') {
    const candidates = value.split(',');
    for (const candidate of candidates) {
      const [a, b] = candidate.split(/[-–—]/);
      const min = toNumber(a);
      const max = toNumber(b);
      if (min !== null && max !== null && min !== max) {
        return { min: Math.min(min, max), max: Math.max(min, max) };
      }
    }
  }

  return null;
}

export async function postTrain(payload) {
  const ds = payload?.dataset_id || getDatasetId();  // sua função util que guarda o id do passo 2
  const spectralRange =
    normalizeSpectralRange(payload?.spectral_range) ||
    normalizeSpectralRange(payload?.spectral_ranges) ||
    normalizeSpectralRange(payload?.ranges);

  const body = {
    dataset_id: ds,
    target_name: payload.target_name ?? payload.target,     // compat
    mode: payload.mode ?? (payload.classification ? "classification" : "regression"),
    n_components: payload.n_components,
    threshold: payload.threshold,
    n_bootstrap: payload.n_bootstrap ?? 0,
    validation_method: payload.validation_method,
    validation_params: payload.validation_params,
    spectral_range: spectralRange ?? undefined,
    preprocess: payload.preprocess,
    preprocess_grid: payload.preprocess_grid,
    sg: payload.sg ?? (Array.isArray(payload.sg_params) && payload.sg_params.length ? payload.sg_params[0] : undefined),
  };
  if (body.n_splits === undefined) {
    const nSplitsCandidate =
      payload?.n_splits ??
      payload?.validation_params?.n_splits ??
      payload?.validation_params?.nSplits ??
      payload?.validation_params?.folds ??
      payload?.validation_params?.k ??
      payload?.validation_params?.nFolds;
    const parsedNSplits = Number(nSplitsCandidate);
    if (Number.isFinite(parsedNSplits) && parsedNSplits > 0) {
      body.n_splits = parsedNSplits;
    }
  }
  return postJSON(`${API_BASE}/train`, body);
}


export async function postPredict(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const api = {
  postColumns,
  postOptimize,
  getOptimizeStatus,
  postReport,
  downloadReport,
  preprocess,
  train,
  predict,
  postPreprocess,
  postPredict,
};

export default api;

