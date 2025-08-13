// === BASE atual (mantive como está) ===
export const API_BASE =
  (window as any).API_BASE || (location.protocol + '//' + location.hostname + ':8000');

// ====== Legado (mantidos) ======
export async function postColumns(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/columns`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postAnalisar(fd: FormData) {
  // LEGADO: se ainda existir esse endpoint no backend, continua.
  // Caso tenha migrado para /train, prefira postTrainForm ou train().
  const res = await fetch(`${API_BASE}/analisar`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postOptimize(file: File, params: Record<string, unknown>) {
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

export async function postReport(payload: unknown) {
  const res = await fetch(`${API_BASE}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}


// ====== NOVO – endpoints JSON do backend ======
// Helpers para garantir que números inválidos virem null (e não NaN/Infinity)
function toNumberOrNull(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  const s = String(v).trim().replace(',', '.');
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

export function normalizeMatrix(rows: Array<Array<number | string | null | undefined>>) {
  return rows.map((r) => r.map(toNumberOrNull));
}

/**
 * /preprocess  -> { X, y? }  -> diag/preview
 * Uso “amigável”: normaliza X automaticamente.
 */
export async function preprocess(
  X: Array<Array<number | string | null | undefined>>,
  y: Array<number | string | null | undefined> | null = null
) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ X: normalizeMatrix(X), y }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * /train -> { X, y, n_components }
 * Uso “amigável”: normaliza X automaticamente.
 */
export async function train(
  X: Array<Array<number | string | null | undefined>>,
  y: Array<number | string | null | undefined>,
  nComponents = 10
) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ X: normalizeMatrix(X), y, n_components: nComponents }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * /predict -> { X }
 * Uso “amigável”: normaliza X automaticamente.
 */
export async function predict(X: Array<Array<number | string | null | undefined>>) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ X: normalizeMatrix(X) }),>>>>>>> main
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


// === Wrappers compatíveis com a outra branch ===
// Aceitam um payload já pronto (sem normalização automática).

export async function postPreprocess(payload: unknown) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),

  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


export async function postTrain(payload: unknown) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),

  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postPredict(payload: unknown) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// NOVO: treino via FormData no endpoint /train (substitui /analisar)
export async function postTrainForm(fd: FormData) {
  const res = await fetch(`${API_BASE}/train`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

