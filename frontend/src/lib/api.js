const API_BASE =
  (typeof window !== 'undefined' && window.API_BASE) ||
  `${location.protocol}//${location.hostname}:8000`;

export async function postJSON(url, body) {
  const fullUrl = url.startsWith("http") ? url : `${API_BASE}${url}`;
  const res = await fetch(fullUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    // keep raw text in case of debugging
  }
  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail ? detail : `HTTP ${res.status}: ${text}`);
  }
  return data;
}

export { API_BASE };
