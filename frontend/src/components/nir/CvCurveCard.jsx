import React, { useMemo } from "react";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function CvCurveCard({ curve, task }) {
  const metric = task === "classification" ? "balanced_accuracy" : "rmsecv";

  const data = useMemo(() => {
    const pts = curve?.points ?? [];
    return pts
      .map(p => ({ k: p.k, y: typeof p[metric] === "number" ? p[metric] : NaN }))
      .filter(d => Number.isFinite(d.y));
  }, [curve, metric]);

  const suggested = curve?.recommended_n_components ?? curve?.recommended_k ?? null;
  const emptyReason = curve?.debug?.reason_if_empty;

  // estado vazio com diagnóstico
  if (!data.length) {
    return (
      <div className="p-4 text-sm text-gray-600">
        <div className="font-medium mb-1">Curva de validação</div>
        <div>Nenhum ponto válido para {metric}.
          {emptyReason ? <> <br/><span className="text-gray-500">{emptyReason}</span></> : null}
        </div>
      </div>
    );
  }

  // domínio sugerido
  const yDomain = task === "classification" ? [0, 1] : [
    Math.min(...data.map(d => d.y)) * 0.98,
    Math.max(...data.map(d => d.y)) * 1.02
  ];

  return (
    <div className="h-64">
      <div className="flex items-center justify-between mb-2">
        <div className="font-medium">Curva de Validação × Nº de Componentes</div>
        <span className="px-2 py-0.5 text-xs rounded bg-emerald-100 text-emerald-700">
          Sugerido: {suggested ?? "—"}
        </span>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="k" label={{ value: "Componentes (PLS)", position: "insideBottom", offset: -5 }} />
          <YAxis domain={yDomain} />
          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} />
          <Line type="monotone" dataKey="y" dot={{ r: 2 }} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
