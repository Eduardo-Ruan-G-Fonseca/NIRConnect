const DEFAULT_API_BASE =
  typeof location !== 'undefined'
    ? `${location.protocol}//${location.hostname}:8000`
    : 'http://localhost:8000';

export const API_BASE =
  typeof window !== 'undefined' && window.API_BASE
    ? window.API_BASE
    : DEFAULT_API_BASE;

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

export async function postOptimize(file, params) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('params', JSON.stringify(params));
  const res = await fetch(`${API_BASE}/optimize`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getOptimizeStatus() {
  const res = await fetch(`${API_BASE}/optimize/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postReport(payload) {
  const res = await fetch(`${API_BASE}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
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

export async function postTrain(payload) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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

// -------------------- FormData variant for /train --------------------
export async function postTrainForm(fd) {
  const res = await fetch(`${API_BASE}/train`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const api = {
  postColumns,
  postAnalisar,
  postOptimize,
  getOptimizeStatus,
  postReport,
  preprocess,
  train,
  predict,
  postPreprocess,
  postTrain,
  postPredict,
  postTrainForm,
};

export default api;

