import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { postReport, downloadReport } from "../../services/api";

export default function Step5Result({ result, onBack, onNew }) {
  const data = result?.data || {};
  const params = result?.params || {};

  const {
    metrics = {},
    analysis_type = "PLS-R",
    y_real = [],
    y_pred = [],
    vip = [],
    features = [],
    scores = [],
    class_mapping = null,
    top_vips = [],
    range_used = "",
    interpretacao_vips = null,
    resumo_interpretativo = "",
    validation_used = null,
    n_splits_effective = null,
    best = null,
    per_class = null,
    curves = null,
  } = data;

  const isClass = analysis_type === "PLS-DA";

  // refs p/ Plotly
  const vipRef = useRef(null);
  const scatterRef = useRef(null);
  const cmRef = useRef(null);

  const [downloading, setDownloading] = useState(false);

  // labels de classes (quando houver)
  const classLabels = useMemo(() => {
    if (class_mapping && typeof class_mapping === "object") {
      return Object.keys(class_mapping)
        .sort((a, b) => Number(a) - Number(b))
        .map((k) => class_mapping[k]);
    }
    // fallback: únicas em y_real
    return [...new Set(y_real || [])];
  }, [class_mapping, y_real]);

  // MÉTRICAS (tabela simples)
  function MetricsTable({ m }) {
    const rows = [];
    for (const [k, v] of Object.entries(m || {})) {
      if (typeof v === "object") continue; // pula blocos (ex: ConfusionMatrix)
      rows.push([k, v]);
    }
    return (
      <table className="min-w-full text-sm">
        <thead>
          <tr>
            <th className="px-4 py-2 border-b text-left">Métrica</th>
            <th className="px-4 py-2 border-b text-left">Valor</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td className="px-4 py-2 border-b font-semibold">{k}</td>
              <td className="px-4 py-2 border-b">
                {Number.isFinite(v) ? Number(v).toFixed(4) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  const renderPlot = (ref, traces, layout, config = { responsive: true }) => {
    if (!ref?.current) return () => {};
    Plotly.newPlot(ref.current, traces, layout, config);
    return () => {
      try {
        Plotly.purge(ref.current);
      } catch {
        /* ignore */
      }
    };
  };

  // VIP chart
  useEffect(() => {
    const trace = {
      x: features,
      y: vip,
      type: "bar",
    };
    return renderPlot(vipRef, [trace], {
      title: "VIP Scores",
      margin: { t: 40, r: 20, l: 40, b: 80 },
    });
  }, [features, vip]);

  // Scatter / Scores / Real vs Pred
  useEffect(() => {
    if (!scatterRef.current) return;

    if (isClass && scores?.length) {
      const xs = scores.map((s) => s[0]);
      const ys = scores.map((s) => s[1] ?? 0);
      const trace = {
        x: xs,
        y: ys,
        mode: "markers",
        type: "scatter",
        marker: { color: y_real },
        text: y_real,
      };
      return renderPlot(
        scatterRef,
        [trace],
        {
          title: "Scores PLS",
          xaxis: { title: "Comp 1" },
          yaxis: { title: "Comp 2" },
        }
      );
    } else if (!isClass && y_real?.length && y_pred?.length) {
      const trace = {
        x: y_real,
        y: y_pred,
        mode: "markers",
        type: "scatter",
      };
      return renderPlot(
        scatterRef,
        [trace],
        {
          title: "y_real vs y_previsto",
          xaxis: { title: "y_real" },
          yaxis: { title: "y_previsto" },
        }
      );
    } else {
      try {
        Plotly.purge(scatterRef.current);
      } catch {
        /* ignore */
      }
      scatterRef.current.innerHTML = "<div class='text-sm text-gray-500'>Sem dados para o gráfico.</div>";
    }
  }, [isClass, scores, y_real, y_pred]);

  // Confusion Matrix (se houver)
  useEffect(() => {
    if (!cmRef.current) return;

    const cm = metrics?.ConfusionMatrix;
    if (isClass && cm && cm.length) {
      const heat = {
        z: cm,
        x: classLabels,
        y: classLabels,
        type: "heatmap",
        colorscale: "YlGnBu",
        text: cm.map((r) => r.map(String)),
        texttemplate: "%{text}",
        textfont: { color: "black" },
      };
      const layout = {
        title: "Matriz de Confusão",
        xaxis: { title: "Predito" },
        yaxis: { title: "Real" },
        margin: { t: 40, r: 20, l: 40, b: 40 },
      };
      return renderPlot(cmRef, [heat], layout);
    } else {
      try {
        Plotly.purge(cmRef.current);
      } catch {
        /* ignore */
      }
      cmRef.current.innerHTML = "";
    }
  }, [isClass, metrics, classLabels]);

  async function handleDownloadPDF() {
    try {
      setDownloading(true);
      const payload = {
        metrics,
        params: {
          ...params,
          analysis_type,
          class_mapping,
        },
        y_real,
        y_pred,
        vip,
        top_vips,
        scores,
        interpretacao_vips,
        resumo_interpretativo,
        validation_used,
        n_splits_effective,
        range_used,
        best,
        per_class,
        curves,
      };
      const { path } = await postReport(payload);
      const blob = await downloadReport(path);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "relatorio.pdf";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
      alert("Falha ao gerar PDF.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Info de validação / resumo */}
      <p className="text-sm text-gray-700">
        {params?.validation_display ? `Validação: ${params.validation_display}` : ""}
      </p>

      {/* MÉTRICAS */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Métricas</h3>
        <MetricsTable m={metrics} />
      </div>

      {/* GRÁFICOS */}
      <div className="grid grid-cols-1 gap-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-2">{isClass ? "Scores PLS" : "Dispersão"}</h3>
          <div ref={scatterRef} className="w-full h-80" />
        </div>

        {isClass && (
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">Matriz de Confusão</h3>
            <div ref={cmRef} className="w-full h-80" />
          </div>
        )}

        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-2">VIP Scores</h3>
          <div ref={vipRef} className="w-full h-80" />
        </div>
      </div>

      {/* INTERPRETAÇÃO */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Interpretação do Modelo</h3>
        <p className="text-sm text-gray-600 mb-2">
          {range_used ? `Faixa utilizada: ${range_used}` : ""}
        </p>
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr>
                <th className="px-4 py-2 border-b text-left">λ (nm)</th>
                <th className="px-4 py-2 border-b text-left">VIP</th>
                <th className="px-4 py-2 border-b text-left">Associação</th>
              </tr>
            </thead>
            <tbody>
              {(top_vips || []).map((row, idx) => (
                <tr key={idx}>
                  <td className="px-4 py-2 border-b">{row.wavelength}</td>
                  <td className="px-4 py-2 border-b">
                    {Number.isFinite(row.vip) ? row.vip.toFixed(2) : row.vip}
                  </td>
                  <td className="px-4 py-2 border-b">{row.label || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {resumo_interpretativo && (
          <p className="text-sm text-gray-700 mt-3">{resumo_interpretativo}</p>
        )}
      </div>

      {/* AÇÕES */}
      <div className="flex gap-2 justify-end">
        <button className="bg-gray-200 px-3 py-2 rounded" onClick={() => onBack?.()}>
          Voltar
        </button>
        <button
          className="bg-[#2e5339] hover:bg-[#305e6b] text-white px-4 py-2 rounded"
          onClick={handleDownloadPDF}
          disabled={downloading}
        >
          {downloading ? "Gerando..." : "Baixar PDF"}
        </button>
        <button
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          onClick={onNew}
        >
          Nova análise
        </button>
      </div>
    </div>
  );
}
