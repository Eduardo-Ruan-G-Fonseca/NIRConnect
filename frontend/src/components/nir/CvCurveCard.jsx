import React, { useMemo } from "react";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function CvCurveCard({ curve, task, recommended }) {
  const data = useMemo(() => {
    if (!curve?.n_components?.length) return [];
    return curve.n_components.map((k, i) => ({
      k,
      accuracy: curve.accuracy?.[i] ?? null,
      balanced_accuracy: curve.balanced_accuracy?.[i] ?? null,
      f1_macro: curve.f1_macro?.[i] ?? null,
      rmsecv: curve.rmsecv?.[i] ?? null,
      r2cv: curve.r2cv?.[i] ?? null,
      auc_macro: curve.auc_macro?.[i] ?? null,
    }));
  }, [curve]);

  if (!curve?.n_components?.length) {
    return <div className="card dashed h-64 flex items-center justify-center"><p>Sem curva de validação.</p></div>;
  }

  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">
        Curva de Validação × Nº de Componentes
        {recommended ? (
          <span className="badge ml-2">Sugerido: k = {recommended}</span>
        ) : null}
      </h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="k" label={{ value: "Componentes (PLS)", position: "insideBottomRight", offset: -5 }} />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          {task === "classification" ? (
            <>
              <Line type="monotone" dataKey="balanced_accuracy" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} name="Balanced Acc." />
              <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" dot={false} strokeWidth={2} isAnimationActive={false} name="Accuracy" />
              <Line type="monotone" dataKey="f1_macro" stroke="#f59e0b" dot={false} strokeWidth={2} isAnimationActive={false} name="F1 macro" />
              <Line type="monotone" dataKey="auc_macro" stroke="#8b5cf6" dot={false} strokeWidth={2} isAnimationActive={false} name="AUC (macro)" />
            </>
          ) : (
            <>
              <Line type="monotone" dataKey="rmsecv" stroke="#ef4444" dot={false} strokeWidth={2} isAnimationActive={false} name="RMSECV" />
              <Line type="monotone" dataKey="r2cv" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} name="R²CV" />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
