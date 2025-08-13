// === BASE atual (mantive como está) ===
export const API_BASE = (window.API_BASE) || (location.protocol + '//' + location.hostname + ':8000');

// ====== Legado (mantidos) ======
export async function postColumns(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/columns`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postAnalisar(fd) {
  // LEGADO: se ainda existir esse endpoint no backend, continua.
  // Se o backend trocou para /train, substitua o uso dessa função pelos métodos "train/preprocess" abaixo nos componentes.
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
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

// ====== NOVO – endpoints JSON do backend ======
// Helpers para garantir que o JSON use null (não NaN/Infinity)
function toNumberOrNull(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  const s = String(v).trim().replace(',', '.');
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}
export function normalizeMatrix(rows) {
  // rows: number[][] | string[][]
  return rows.map(r => r.map(toNumberOrNull));
}

/**
 * /preprocess  -> { X, y? }  -> diag/preview
 */
export async function preprocess(X, y = null) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ X: normalizeMatrix(X), y })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * /train -> { X, y, n_components }
 */
export async function train(X, y, nComponents = 10) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ X: normalizeMatrix(X), y, n_components: nComponents })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * /predict -> { X }
 */
export async function predict(X) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ X: normalizeMatrix(X) })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// NOVO: treino via FormData no endpoint /train (substitui /analisar)
export async function postTrainForm(fd) {
  const res = await fetch(`${API_BASE}/train`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
