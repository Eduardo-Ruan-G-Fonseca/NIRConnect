// frontend/src/components/nir/ConfusionMatrixCard.jsx
import React from "react";

function Cell({ value, frac }) {
  const bg = Number.isFinite(frac) ? `rgba(16, 185, 129, ${Math.min(1, Math.max(0.05, frac))})` : "transparent";
  return (
    <td style={{ background: bg }} className="text-center">{value}</td>
  );
}

export default function ConfusionMatrixCard({ cm }) {
  if (!cm?.labels?.length || !cm?.matrix?.length) {
    return (
      <div className="card dashed h-64 flex items-center justify-center">
        <p>Não disponível para este modo/validação.</p>
      </div>
    );
  }

  const labels = cm.labels;
  const M = cm.matrix;
  const N = cm.normalized;

  return (
    <div className="card p-4" style={{minHeight: 420}}>
      <h3 className="card-title mb-3">Matriz de Confusão</h3>
      <div className="overflow-x-auto">
        <table className="table table-sm">
          <thead>
            <tr>
              <th>Verdadeiro \ Predito</th>
              {labels.map((l, j) => <th key={j} className="text-center">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {M.map((row, i) => (
              <tr key={i}>
                <th>{labels[i]}</th>
                {row.map((v, j) => (
                  <Cell key={j} value={v} frac={N ? N[i][j] : null} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
