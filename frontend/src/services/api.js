
import { postJSON } from "../api/http";

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

export async function postOptimize(payload) {
  return postJSON(`${API_BASE}/optimize`, payload);
}

export async function getOptimizeStatus() {
  const res = await fetch(`${API_BASE}/optimize/status`);
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


export async function postTrain(payload) {
  return postJSON(`${API_BASE}/train`, payload);
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

