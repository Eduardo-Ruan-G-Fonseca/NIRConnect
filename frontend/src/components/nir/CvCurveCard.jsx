import React, { useMemo } from "react";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

/**
 * props.curve: objeto retornado pelo backend
 * props.task : "classification" | "regression"
 * props.suggestedK: número sugerido (opcional)
 */
export default function CvCurveCard({ curve, task, suggestedK }) {
  const data = useMemo(() => {
    if (!curve) return [];
    if (Array.isArray(curve.points) && curve.points.length) return curve.points;

    // fallback: construir a lista a partir dos vetores
    const ks = curve.n_components || [];
    const acc  = curve.accuracy || [];
    const bacc = curve.balanced_accuracy || [];
    const f1m  = curve.f1_macro || [];
    const auc  = curve.auc_macro || [];
    const rmse = curve.rmsecv || [];
    const r2   = curve.r2cv || [];
    return ks.map((k, i) => ({
      k,
      accuracy: acc[i] ?? null,
      balanced_accuracy: bacc[i] ?? null,
      f1_macro: f1m[i] ?? null,
      auc_macro: auc[i] ?? null,
      rmsecv: rmse[i] ?? null,
      r2cv: r2[i] ?? null,
    }));
  }, [curve]);

  // há pelo menos UMA série com algum valor?
  const hasClsSeries = useMemo(() => {
    if (task !== "classification") return false;
    return data.some(d => d?.balanced_accuracy != null || d?.accuracy != null || d?.f1_macro != null || d?.auc_macro != null);
  }, [data, task]);

  const hasRegSeries = useMemo(() => {
    if (task === "classification") return false;
    return data.some(d => d?.rmsecv != null || d?.r2cv != null);
  }, [data, task]);

  return (
    <div className="card p-4" id="cv-curve">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="card-title">Curva de Validação × Nº de Componentes</h3>
        {Number.isFinite(suggestedK) && (
          <span className="ml-2 text-xs px-2 py-1 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
            Sugerido: k = {suggestedK}
          </span>
        )}
      </div>

      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" label={{ value: "Componentes (PLS)", position: "insideBottomRight", offset: -10 }} />
            <YAxis domain={["auto", "auto"]} />
            <Tooltip />
            <Legend />

            {task === "classification" ? (
              hasClsSeries ? (
                <>
                  <Line type="monotone" dataKey="balanced_accuracy" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} name="Balanced Acc." />
                  <Line type="monotone" dataKey="accuracy"            stroke="#3b82f6"  strokeWidth={2} dot={false} isAnimationActive={false} name="Accuracy" />
                  <Line type="monotone" dataKey="f1_macro"            stroke="#f59e0b"  strokeWidth={2} dot={false} isAnimationActive={false} name="F1 macro" />
                  <Line type="monotone" dataKey="auc_macro"           stroke="#8b5cf6"  strokeWidth={2} dot={false} isAnimationActive={false} name="AUC (macro)" />
                </>
              ) : null
            ) : hasRegSeries ? (
              <>
                <Line type="monotone" dataKey="rmsecv" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} name="RMSECV" />
                <Line type="monotone" dataKey="r2cv"   stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} name="R²CV" />
              </>
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {((task === "classification" && !hasClsSeries) || (task !== "classification" && !hasRegSeries)) && (
        <div className="text-sm text-gray-500 mt-3">
          Sem pontos válidos para plotar a curva. Verifique o método de validação, classes muito raras ou número de componentes.
        </div>
      )}
    </div>
  );
}

