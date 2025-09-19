export function extractVip(res) {
  if (res?.vip?.scores?.length) {
    const wl = res.vip.wavelengths || res.wavelengths || [];
    return res.vip.scores.map((s, i) => ({ wavelength: Number(wl?.[i] ?? i), score: Number(s) }))
      .filter(d => Number.isFinite(d.score))
      .sort((a,b)=>b.score-a.score);
  }
  if (Array.isArray(res?.vips) && res.vips.length) {
    return res.vips.map(d => ({ wavelength: Number(d.wavelength), score: Number(d.score) }))
      .filter(d => Number.isFinite(d.score))
      .sort((a,b)=>b.score-a.score);
  }
  return [];
}
export function extractConfusion(res) {
  if (res?.confusion_matrix?.matrix) {
    return {
      labels: res.confusion_matrix.labels || [],
      matrix: res.confusion_matrix.matrix || [],
      normalized: res.confusion_matrix.normalized || null,
    };
  }
  if (res?.cm && res?.labels) return { labels: res.labels, matrix: res.cm, normalized: null };
  return null;
}
export function normalizeTrainResult(res) {
  return {
    task: res?.task || "classification",
    metrics: res?.metrics || {},
    per_class: res?.per_class || null,     // <- NOVO (tabela por classe)
    cv: res?.cv || {},
    cv_metrics: res?.cv_metrics || null,
    cv_curve: res?.cv_curve || null,
    recommended_n_components: res?.recommended_n_components ?? null,
    meta: res?.meta || null,
    train_time_seconds: res?.train_time_seconds ?? null,
    vip: extractVip(res),
    cm: extractConfusion(res),
    wavelengths: res?.wavelengths || [],
    latent: res?.latent || null,
    oof: res?.oof || null,
    residuals: res?.residuals || null,
    influence: res?.influence || null,
    distributions: res?.distributions || null,
    predictions: res?.predictions || null,
  };
}
