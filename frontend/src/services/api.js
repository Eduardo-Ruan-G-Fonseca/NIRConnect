
const DEFAULT_API_BASE =
  typeof location !== 'undefined'
    ? location.protocol + '//' + location.hostname + ':8000'
    : 'http://localhost:8000';

export const API_BASE =
  typeof window !== 'undefined' && window.API_BASE
    ? window.API_BASE
    : DEFAULT_API_BASE;


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

// ---- New PLS pipeline endpoints ----
export async function postPreprocess(payload) {
  const res = await fetch(`${API_BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)

  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function postTrain(payload) {
  if (payload instanceof FormData) {
    let res = await fetch(`${API_BASE}/train`, { method: 'POST', body: payload });
    if (!res.ok) {
      res = await fetch(`${API_BASE}/analisar`, { method: 'POST', body: payload });
      if (!res.ok) {
        res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: payload });
      }
    }
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } else {
    const res = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
}

export async function postPredict(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)

  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const api = {
  postColumns,
  postAnalisar,
  postOptimize,
  getOptimizeStatus,
  postReport,
  postPreprocess,
  postTrain,
  postPredict,
};

export default api;
