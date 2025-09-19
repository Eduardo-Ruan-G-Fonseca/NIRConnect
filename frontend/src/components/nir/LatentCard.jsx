import React, { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function LatentCard({ latent, labels }) {
  const scoreMatrix = useMemo(() => (
    Array.isArray(latent?.scores) ? latent.scores : []
  ), [latent?.scores]);
  const data = useMemo(() => scoreMatrix.map((row, i) => ({
    lv1: row[0] ?? 0, lv2: row[1] ?? 0, label: labels?.[i] ?? ""
  })), [scoreMatrix, labels]);

  if (!scoreMatrix.length) {
    return <div className="card dashed h-64 flex items-center justify-center"><p>Sem variáveis latentes.</p></div>;
  }

  // Paleta simples por classe
  const uniq = Array.from(new Set(data.map(d => d.label)));
  const colors = ["#2563eb","#16a34a","#f59e0b","#ef4444","#7c3aed","#0ea5e9","#84cc16","#d946ef","#fb7185","#22c55e"];

  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">Latentes (Scores LV1 × LV2)</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="col-span-1">
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="lv1" name="LV1" />
              <YAxis type="number" dataKey="lv2" name="LV2" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {uniq.map((c, idx) => (
                <Scatter key={c} name={c} data={data.filter(d => d.label === c)} fill={colors[idx % colors.length]} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="col-span-1">
          <div className="rounded border p-3">
            <h3 className="font-medium mb-2">R² Cumulativo</h3>
            <div style={{maxHeight: 260, overflowY: "auto"}}>
              <table className="table table-sm">
                <thead><tr><th>Comp</th><th>R²X (cum)</th><th>R²Y (cum)</th></tr></thead>
                <tbody>
                  {(latent.r2x_cum || []).map((rx, i) => (
                    <tr key={i}><td>{i+1}</td><td>{rx.toFixed(4)}</td><td>{(latent.r2y_cum?.[i] ?? 0).toFixed(4)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
