import React, { useMemo } from "react";

export default function OptimizeTrialsCard({ trials, task }) {
  const rows = useMemo(() => {
    if (!Array.isArray(trials)) return [];
    const metricKey = task === "classification" ? "balanced_accuracy" : "rmsecv";
    return trials
      .map(t => {
        const m = t?.cv_metrics?.[metricKey];
        const score = typeof m === "number" ? m : (task === "classification" ? -Infinity : Infinity);
        return { ...t, _score: score };
      })
      .sort((a, b) => {
        return task === "classification" ? b._score - a._score : a._score - b._score;
      });
  }, [trials, task]);

  if (!rows.length) {
    return null;
  }

  return (
    <div className="rounded border p-3">
      <h3 className="font-medium mb-2">Busca de Parâmetros</h3>
      <div style={{ maxHeight: 260, overflowY: "auto" }}>
        <table className="table table-sm">
          <thead>
            <tr>
              <th>#</th>
              <th>Preprocess</th>
              <th>SG</th>
              <th>k</th>
              <th>Vars</th>
              <th>{task === "classification" ? "BACC" : "RMSECV"}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t, idx) => (
              <tr key={idx}>
                <td>{idx + 1}</td>
                <td>{(t.params?.preprocess || []).join("+") || "—"}</td>
                <td>{t.params?.sg ? t.params.sg.join(",") : "—"}</td>
                <td>{t.params?.n_components}</td>
                <td>{t.params?.mask_vars}</td>
                <td>{Number.isFinite(t._score) ? t._score.toFixed(4) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
