// --- dataset id helpers ---
const DS_KEY = "dataset_id";

export function setDatasetId(id) {
  try {
    localStorage.setItem(DS_KEY, id || "");
  } catch {
    /* ignore */
  }
}

export function getDatasetId() {
  try {
    return localStorage.getItem(DS_KEY) || null;
  } catch {
    /* ignore */
    return null;
  }
}

export function clearDatasetId() {
  try {
    localStorage.removeItem(DS_KEY);
  } catch {
    /* ignore */
  }
}

// --- JSON client ---
export async function postJSON(url, body = {}, opts = {}) {
  // anexa dataset_id se ainda não tiver no body
  const ds = body.dataset_id ?? getDatasetId();
  const payload = ds ? { ...body, dataset_id: ds } : { ...body };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    body: JSON.stringify(payload),
    signal: opts.signal,
  });

  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = { raw: text };
  }

  if (!res.ok) {
    const msg = json?.detail || json?.message || `Erro ${res.status}`;
    const hint = json?.hint ? ` ${json.hint}` : "";
    throw new Error(`${msg}${hint}`);
  }
  return json;
}
