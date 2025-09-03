import React from "react";

export default function PerClassMetricsCard({ perClass }) {
  if (!perClass?.length) return null;
  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">Métricas por Classe</h3>
      <div className="overflow-x-auto" style={{maxHeight: 280, overflowY:"auto"}}>
        <table className="table table-sm">
          <thead><tr><th>Classe</th><th>Precisão</th><th>Recall</th><th>F1</th><th>Suporte</th></tr></thead>
          <tbody>
            {perClass.map((r, i) => (
              <tr key={i}>
                <td>{r.label}</td>
                <td>{r.precision.toFixed(3)}</td>
                <td>{r.recall.toFixed(3)}</td>
                <td>{r.f1.toFixed(3)}</td>
                <td>{r.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
