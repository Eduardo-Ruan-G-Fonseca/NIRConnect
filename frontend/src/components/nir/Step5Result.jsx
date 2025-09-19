import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { postReport, downloadReport } from "../../services/api";
import { normalizeTrainResult } from "../../services/normalizeTrainResult";

function MetricsTable({ title, metrics }) {
  const entries = Object.entries(metrics || {}).filter(([, v]) => v !== null && v !== undefined);
  return (
    <div className="card p-4 space-y-3">
      <h3 className="card-title">{title}</h3>
      {entries.length ? (
        <table className="table table-sm">
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k}>
                <th className="font-medium text-gray-600">{k}</th>
                <td>{Number.isFinite(v) ? Number(v).toFixed(4) : String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-sm text-gray-500">Sem dados disponíveis.</p>
      )}
    </div>
  );
}

function PerClassTable({ perClass }) {
  if (!Array.isArray(perClass) || !perClass.length) {
    return null;
  }
  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">Desempenho por Classe</h3>
      <div className="overflow-x-auto">
        <table className="table table-sm">
          <thead>
            <tr>
              <th>Classe</th>
              <th>Precisão</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Suporte</th>
            </tr>
          </thead>
          <tbody>
            {perClass.map((row, idx) => (
              <tr key={`${row.label}-${idx}`}>
                <td>{row.label}</td>
                <td>{Number(row.precision).toFixed(3)}</td>
                <td>{Number(row.recall).toFixed(3)}</td>
                <td>{Number(row.f1).toFixed(3)}</td>
                <td>{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Step5Result({ result, onBack, onNew }) {
  const rawData = useMemo(() => result?.data || result || {}, [result]);
  const data = useMemo(() => normalizeTrainResult(rawData), [rawData]);

  const {
    task,
    metrics,
    cv_metrics: cvMetrics,
    residuals,
    influence,
    distributions,
    predictions,
    per_class: perClass,
  } = data;

  const residualRef = useRef(null);
  const stdResidualRef = useRef(null);
  const leverageRef = useRef(null);
  const histogramRef = useRef(null);

  const [downloading, setDownloading] = useState(false);

  const residualOutliers = residuals?.outliers?.length ?? 0;
  const highLeverage = influence?.high_leverage?.length ?? 0;
  const hotellingOutliers = influence?.hotelling_outliers?.length ?? 0;

  const leverageThreshold = influence?.leverage_threshold ?? null;
  const stdThreshold = residuals?.std_threshold ?? null;
  const hotellingThreshold = influence?.hotelling_t2_threshold ?? null;

  const sampleIndex = useMemo(() => {
    const base = residuals?.sample_index ?? predictions?.sample_index ?? [];
    return Array.isArray(base) ? base : [];
  }, [residuals?.sample_index, predictions?.sample_index]);

  const renderPlot = (ref, traces, layout) => {
    if (!ref?.current) return () => {};
    if (!Array.isArray(traces) || !traces.length) {
      ref.current.innerHTML = "<div class='text-sm text-gray-500'>Sem dados para exibir.</div>";
      return () => {};
    }
    Plotly.newPlot(ref.current, traces, layout, { responsive: true, displaylogo: false });
    return () => {
      try {
        Plotly.purge(ref.current);
      } catch {
        /* ignore */
      }
    };
  };

  useEffect(() => {
    if (!residuals?.predicted?.length || !residuals?.raw?.length) {
      if (residualRef.current) {
        residualRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem dados de resíduos.</div>";
      }
      return () => {};
    }

    const preds = residuals.predicted;
    const res = residuals.raw;
    const labels = residuals.y_true || [];
    const predsLabel = residuals.y_pred || [];
    const idxMap = new Map(sampleIndex.map((id, pos) => [id, pos]));
    const outlierSet = new Set(residuals.outliers || []);

    const buildTrace = (ids, name, color) => {
      const xs = ids.map((id) => preds[idxMap.get(id)]);
      const ys = ids.map((id) => res[idxMap.get(id)]);
      const text = ids.map((id) => {
        const pos = idxMap.get(id);
        const lbl = labels[pos] ?? "";
        const predLbl = predsLabel[pos] ?? "";
        return `Amostra ${id}<br>Real: ${lbl}<br>Previsto: ${predLbl}`;
      });
      return {
        x: xs,
        y: ys,
        mode: "markers",
        type: "scatter",
        name,
        text,
        hovertemplate: "%{text}<br>Previsto: %{x:.3f}<br>Resíduo: %{y:.3f}<extra></extra>",
        marker: { size: 8, color },
      };
    };

    const outlierIds = sampleIndex.filter((id) => outlierSet.has(id));
    const inlierIds = sampleIndex.filter((id) => !outlierSet.has(id));

    const traces = [];
    if (inlierIds.length) traces.push(buildTrace(inlierIds, "Amostras", "#1f77b4"));
    if (outlierIds.length) traces.push(buildTrace(outlierIds, "Outliers", "#d62728"));
    traces.push({
      x: [Math.min(...preds), Math.max(...preds)],
      y: [0, 0],
      mode: "lines",
      name: "Resíduo = 0",
      line: { color: "#888", dash: "dash" },
      hoverinfo: "skip",
    });

    return renderPlot(residualRef, traces, {
      title: "Resíduos vs. Previsto",
      xaxis: { title: task === "classification" ? "Probabilidade prevista" : "Valor previsto" },
      yaxis: { title: "Resíduo" },
      margin: { t: 40, r: 20, l: 50, b: 50 },
    });
  }, [residuals, sampleIndex, task]);

  useEffect(() => {
    if (!residuals?.standardized?.length) {
      if (stdResidualRef.current) {
        stdResidualRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem dados padronizados.</div>";
      }
      return () => {};
    }

    const stdValues = residuals.standardized;
    const idx = sampleIndex;
    if (!idx.length) {
      if (stdResidualRef.current) {
        stdResidualRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem índice de amostras.</div>";
      }
      return () => {};
    }
    const outlierSet = new Set(residuals.outliers || []);
    const traceBase = {
      x: idx,
      y: stdValues,
      mode: "markers",
      type: "scatter",
      name: "Resíduo padronizado",
      marker: { color: "#2ca02c", size: 8 },
      hovertemplate: "Amostra %{x}<br>Valor: %{y:.2f}<extra></extra>",
    };
    const traceOut = {
      x: idx.filter((id) => outlierSet.has(id)),
      y: idx.filter((id) => outlierSet.has(id)).map((id) => stdValues[idx.indexOf(id)]),
      mode: "markers",
      type: "scatter",
      name: "Outliers",
      marker: { color: "#d62728", size: 10 },
      hovertemplate: "Amostra %{x}<br>Valor: %{y:.2f}<extra></extra>",
    };

    const threshold = Number(stdThreshold) || 3;

    const traces = [traceBase];
    if (traceOut.x.length) traces.push(traceOut);
    traces.push({
      x: [Math.min(...idx), Math.max(...idx)],
      y: [threshold, threshold],
      mode: "lines",
      name: `+${threshold.toFixed(1)}σ`,
      line: { color: "#ff7f0e", dash: "dot" },
      hoverinfo: "skip",
    });
    traces.push({
      x: [Math.min(...idx), Math.max(...idx)],
      y: [-threshold, -threshold],
      mode: "lines",
      name: `-${threshold.toFixed(1)}σ`,
      line: { color: "#ff7f0e", dash: "dot" },
      hoverinfo: "skip",
    });

    return renderPlot(stdResidualRef, traces, {
      title: "Resíduos Padronizados",
      xaxis: { title: "Índice da amostra" },
      yaxis: { title: "Resíduo padronizado" },
      margin: { t: 40, r: 20, l: 50, b: 50 },
    });
  }, [residuals, sampleIndex, stdThreshold]);

  useEffect(() => {
    if (!influence?.leverage?.length) {
      if (leverageRef.current) {
        leverageRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem dados de leverage.</div>";
      }
      return () => {};
    }

    const lev = influence.leverage;
    const idx = sampleIndex.length === lev.length ? sampleIndex : lev.map((_, i) => i);
    if (!idx.length) {
      if (leverageRef.current) {
        leverageRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem índice de amostras.</div>";
      }
      return () => {};
    }
    const outSet = new Set(influence.high_leverage || []);
    const traceBase = {
      x: idx,
      y: lev,
      mode: "markers",
      type: "scatter",
      name: "Leverage",
      marker: { color: "#17becf", size: 8 },
      hovertemplate: "Amostra %{x}<br>Leverage: %{y:.3f}<extra></extra>",
    };
    const traceOut = {
      x: idx.filter((id) => outSet.has(id)),
      y: idx.filter((id) => outSet.has(id)).map((id) => lev[idx.indexOf(id)]),
      mode: "markers",
      type: "scatter",
      name: "Leverage alto",
      marker: { color: "#d62728", size: 10 },
      hovertemplate: "Amostra %{x}<br>Leverage: %{y:.3f}<extra></extra>",
    };

    const threshold = Number(leverageThreshold) || 0;

    const traces = [traceBase];
    if (traceOut.x.length) traces.push(traceOut);
    if (threshold > 0) {
      traces.push({
        x: [Math.min(...idx), Math.max(...idx)],
        y: [threshold, threshold],
        mode: "lines",
        name: `Limite (${threshold.toFixed(3)})`,
        line: { color: "#ff7f0e", dash: "dash" },
        hoverinfo: "skip",
      });
    }

    return renderPlot(leverageRef, traces, {
      title: "Leverage (Hotelling T²)",
      xaxis: { title: "Índice da amostra" },
      yaxis: { title: "Leverage" },
      margin: { t: 40, r: 20, l: 50, b: 50 },
    });
  }, [influence, sampleIndex, leverageThreshold]);

  useEffect(() => {
    const prob = distributions?.probabilities;
    const predHist = distributions?.predicted;

    if (task === "classification" && prob && Object.keys(prob).length) {
      const traces = Object.entries(prob)
        .map(([label, hist]) => {
          const edges = hist?.bin_edges || [];
          const counts = hist?.counts || [];
          if (!edges.length || !counts.length) return null;
          const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
          return {
            type: "bar",
            name: label,
            x: centers,
            y: counts,
            opacity: 0.75,
          };
        })
        .filter(Boolean);
      return renderPlot(histogramRef, traces, {
        barmode: "stack",
        title: "Distribuição das Probabilidades Previstas",
        xaxis: { title: "Probabilidade", range: [0, 1] },
        yaxis: { title: "Frequência" },
        margin: { t: 40, r: 20, l: 50, b: 50 },
      });
    }

    if (task === "regression" && predHist?.bin_edges?.length) {
      const edges = predHist.bin_edges;
      const counts = predHist.counts || [];
      const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
      const trace = {
        type: "bar",
        x: centers,
        y: counts,
        marker: { color: "#1f77b4" },
      };
      return renderPlot(histogramRef, [trace], {
        title: "Distribuição das Previsões",
        xaxis: { title: "Valor previsto" },
        yaxis: { title: "Frequência" },
        margin: { t: 40, r: 20, l: 50, b: 50 },
      });
    }

    if (histogramRef.current) {
      histogramRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem histogramas disponíveis.</div>";
    }
    return () => {};
  }, [distributions, task]);

  async function handleDownloadPDF() {
    try {
      setDownloading(true);
      const metricPayload = {
        ...Object.fromEntries(Object.entries(metrics || {}).map(([k, v]) => [`Treino - ${k}`, v])),
        ...Object.fromEntries(Object.entries(cvMetrics || {}).map(([k, v]) => [`Validação - ${k}`, v])),
      };
      const curvesRaw = Array.isArray(rawData?.curves)
        ? rawData.curves
        : rawData?.curves
        ? [rawData.curves]
        : [];
      if (!curvesRaw.length && data?.cv_curve?.points) {
        curvesRaw.push(data.cv_curve);
      }
      const payload = {
        metrics: metricPayload,
        params: result?.params || {},
        validation_used: rawData?.validation_used || data?.cv?.validation?.method,
        n_splits_effective: rawData?.n_splits_effective || data?.cv?.validation?.splits,
        range_used: rawData?.range_used,
        best: rawData?.best,
        per_class: perClass,
        curves: curvesRaw,
      };
      const resp = await postReport(payload);
      if (resp?.path) {
        const blob = await downloadReport(resp.path);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "relatorio_nir.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      const msg = err?.message || "Falha ao gerar relatório.";
      alert(msg);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-[#2e5339]">Resultado Final</h2>
          <p className="text-sm text-gray-600">
            {task === "classification"
              ? "Modelo PLS-DA com diagnóstico completo de resíduos e influência."
              : "Modelo PLS-R com análise de resíduos e previsão."}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleDownloadPDF}
            className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg shadow-sm disabled:opacity-60"
            disabled={downloading}
          >
            {downloading ? <i className="fas fa-spinner fa-spin mr-2" /> : <i className="fas fa-file-download mr-2" />}
            Baixar relatório PDF
          </button>
          <button
            onClick={onNew}
            className="bg-emerald-700 hover:bg-emerald-800 text-white py-2 px-4 rounded-lg"
          >
            Novo projeto
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            title: "Resíduos extremos",
            value: residualOutliers,
            description: stdThreshold
              ? `|resíduo padronizado| > ${Number(stdThreshold).toFixed(1)}`
              : "|resíduo padronizado| > 3",
          },
          {
            title: "Leverage elevado",
            value: highLeverage,
            description: leverageThreshold
              ? `Leverage > ${Number(leverageThreshold).toFixed(3)}`
              : "Acima do limite recomendado",
          },
          {
            title: "Hotelling T²",
            value: hotellingOutliers,
            description: hotellingThreshold
              ? `Acima de ${Number(hotellingThreshold).toFixed(2)}`
              : "Acima do percentil 95%",
          },
        ].map((card) => (
          <div
            key={card.title}
            className={`card p-4 border ${card.value ? "border-red-300 bg-red-50" : "border-green-200 bg-green-50"}`}
          >
            <h3 className="text-lg font-semibold text-gray-700">{card.title}</h3>
            <p className="text-3xl font-bold text-gray-900">{card.value}</p>
            <p className="text-sm text-gray-600">{card.description}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricsTable title="Métricas de Treino" metrics={metrics} />
        <MetricsTable title="Métricas de Validação" metrics={cvMetrics} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-4">
          <div ref={residualRef} style={{ minHeight: 320 }} />
        </div>
        <div className="card p-4">
          <div ref={stdResidualRef} style={{ minHeight: 320 }} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-4">
          <div ref={leverageRef} style={{ minHeight: 320 }} />
        </div>
        <div className="card p-4">
          <div ref={histogramRef} style={{ minHeight: 320 }} />
        </div>
      </div>

      <PerClassTable perClass={perClass} />

      <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-between">
        <button onClick={onBack} className="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-lg">
          Voltar
        </button>
        <div className="text-sm text-gray-600">
          {predictions?.class_labels?.length
            ? `Classes: ${predictions.class_labels.join(", ")}`
            : task === "regression"
            ? "Modelo de regressão"
            : ""}
        </div>
      </div>
    </div>
  );
}
