// frontend/src/services/normalizeTrainResult.js
export function extractVip(res) {
  // tenta { vip: {wavelengths, scores} }
  if (res?.vip?.scores?.length) {
    const wl = res.vip.wavelengths || res.wavelengths || [];
    return res.vip.scores.map((score, i) => ({
      wavelength: Number(wl?.[i] ?? i),
      score: Number(score),
    })).filter(d => Number.isFinite(d.score));
  }
  // tenta [{ wavelength, score }]
  if (Array.isArray(res?.vips) && res.vips.length) {
    return res.vips.map(d => ({
      wavelength: Number(d.wavelength),
      score: Number(d.score),
    })).filter(d => Number.isFinite(d.score));
  }
  // nada
  return [];
}

export function extractConfusion(res) {
  if (res?.confusion_matrix?.matrix) {
    const labels = res.confusion_matrix.labels || [];
    const matrix = res.confusion_matrix.matrix || [];
    const normalized = res.confusion_matrix.normalized || null;
    return { labels, matrix, normalized };
  }
  // formatos antigos opcionais
  if (res?.cm && res?.labels) {
    return { labels: res.labels, matrix: res.cm, normalized: null };
  }
  return null;
}

export function normalizeTrainResult(res) {
  const vip = extractVip(res)
    .sort((a, b) => b.score - a.score);
  const cm = extractConfusion(res);

  return {
    task: res?.task || "classification",
    metrics: res?.metrics || {},
    cv: res?.cv || {},
    cv_curve: res?.cv_curve || null,
    vip,
    cm,
    wavelengths: res?.wavelengths || [],
    latent: res?.latent || null,
    oof: res?.oof || null,
  };
}
