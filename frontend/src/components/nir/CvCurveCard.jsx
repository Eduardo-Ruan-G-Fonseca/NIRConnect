import React, { useMemo } from "react";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function CvCurveCard({ curve, task }) {
  if (!curve?.n_components?.length) {
    return <div className="card dashed h-64 flex items-center justify-center"><p>Sem curva de validação.</p></div>;
  }
  const data = useMemo(() => {
    return curve.n_components.map((k, i) => ({
      k,
      accuracy: curve.accuracy?.[i] ?? null,
      balanced_accuracy: curve.balanced_accuracy?.[i] ?? null,
      f1_macro: curve.f1_macro?.[i] ?? null,
      rmsecv: curve.rmsecv?.[i] ?? null,
      r2cv: curve.r2cv?.[i] ?? null,
    }));
  }, [curve]);

  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">Curva de Validação × Nº de Componentes</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="k" label={{ value: "Componentes (PLS)", position: "insideBottomRight", offset: -5 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          {task === "classification" ? (
            <>
              <Line type="monotone" dataKey="balanced_accuracy" stroke="#10b981" dot={false} name="Balanced Acc." />
              <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" dot={false} name="Accuracy" />
              <Line type="monotone" dataKey="f1_macro" stroke="#f59e0b" dot={false} name="F1 macro" />
            </>
          ) : (
            <>
              <Line type="monotone" dataKey="rmsecv" stroke="#ef4444" dot={false} name="RMSECV" />
              <Line type="monotone" dataKey="r2cv" stroke="#10b981" dot={false} name="R²CV" />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
