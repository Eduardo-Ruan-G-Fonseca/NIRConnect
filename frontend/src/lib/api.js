const API_BASE =
  (typeof window !== 'undefined' && window.API_BASE) ||
  `${location.protocol}//${location.hostname}:8000`;

export async function postJSON(url, payload) {
  const fullUrl = url.startsWith('http') ? url : `${API_BASE}${url}`;
  const res = await fetch(fullUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { message: text }; }
  if (!res.ok) {
    const msg = data.detail?.[0]?.msg ?? data.detail?.msg ?? data.message ?? 'Erro na requisição';
    throw new Error(msg);
  }
  return data;
}

export { API_BASE };
