import { useEffect, useMemo, useState } from "react";
import { postOptimize, getOptimizeStatus, postTrainForm } from "../../services/api";

// rótulos bonitinhos
const PREP_LABEL = {
  none: "Nenhum",
  snv: "SNV",
  msc: "MSC",
  sg1: "1ª Derivada",
  sg2: "2ª Derivada",
  minmax: "Min–Max",
  zscore: "Z-score",
};

export default function Step4Decision({ file, step2, result, onBack, onContinue }) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pollId, setPollId] = useState(null);
  const [optResults, setOptResults] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");

  const isClass = !!step2?.classification;
  const metricLabel = isClass ? "Accuracy" : "RMSECV";

  // tenta extrair [min,max] de uma string "950-1650" etc
  const spectralRange = useMemo(() => {
    const s = result?.params?.ranges || "";
    const m = s.match(/(-?\d+(\.\d+)?)[^\d]+(-?\d+(\.\d+)?)/);
    return m ? [parseFloat(m[1]), parseFloat(m[3])] : undefined;
  }, [result]);

  // lista de métodos já usados no step3 (para otimizar em cima deles)
  const preprocessingList = useMemo(() => {
    const steps = result?.params?.preprocess_steps || [];
    const methods = steps.map((p) => p.method);
    return methods.length ? methods : ["none"];
  }, [result]);

  useEffect(() => {
    return () => {
      if (pollId) clearInterval(pollId);
    };
  }, [pollId]);

  async function startOptimize() {
    setError("");
    setRunning(true);
    setProgress(0);
    setOptResults(null);
    setSelected(null);

    // inicia polling do /optimize/status
    const id = setInterval(async () => {
      try {
        const s = await getOptimizeStatus();
        const pct = s.total ? Math.round((s.current / s.total) * 100) : 0;
        setProgress(pct);
      } catch {
        // ignora erros de polling
      }
    }, 1000);
    setPollId(id);

    try {
      console.debug('[Step4] params recebidos =', result?.params);
      const payload = {
        target: step2.target,
        validation_method: step2.validation_method,
        n_components: parseInt(step2.n_components, 10),
        n_bootstrap: parseInt(step2.n_bootstrap || 0, 10),
        folds:
          step2.validation_method === "KFold"
            ? step2.validation_params?.n_splits || 5
            : undefined,
        analysis_mode: step2.classification ? "PLS-DA" : "PLS-R",
        spectral_range: spectralRange,
        preprocessing_methods: preprocessingList,
      };

      const data = await postOptimize(file, payload);
      setOptResults(data.results || []);
      setProgress(100);
    } catch (e) {
      setError(typeof e === "string" ? e : "Falha na otimização.");
    } finally {
      setRunning(false);
      if (pollId) clearInterval(pollId);
      setPollId(null);
    }
  }

  function sortResults(rs) {
    const arr = [...rs];
    return arr.sort((a, b) => {
      const aval = isClass ? -(a?.val_metrics?.Accuracy || 0) : (a?.RMSECV ?? 1e9);
      const bval = isClass ? -(b?.val_metrics?.Accuracy || 0) : (b?.RMSECV ?? 1e9);
      return aval - bval;
    });
  }

  function renderTable(rs) {
    const sorted = sortResults(rs);
    return (
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="px-4 py-2 border-b text-left">Pré-processamento</th>
              <th className="px-4 py-2 border-b text-left">n_components</th>
              <th className="px-4 py-2 border-b text-left">{metricLabel}</th>
              <th className="px-4 py-2 border-b text-left">R²</th>
              <th className="px-4 py-2 border-b text-left">Validação</th>
              <th className="px-4 py-2 border-b text-left">Faixa usada</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const prep = PREP_LABEL[r.preprocess] || r.preprocess;
              const metric = isClass
                ? (r?.val_metrics?.Accuracy ?? 0).toFixed(3)
                : (r?.RMSECV ?? 0).toFixed(3);
              const r2 =
                r?.val_metrics?.R2 !== undefined
                  ? Number(r.val_metrics.R2).toFixed(3)
                  : "-";
              const valMethod = r?.validation?.method || "-";
              const range =
                r?.wl_used?.length
                  ? `${r.wl_used[0]}-${r.wl_used[r.wl_used.length - 1]}`
                  : "-";
              const active =
                selected &&
                selected.preprocess === r.preprocess &&
                selected.n_components === r.n_components;

              return (
                <tr
                  key={`${r.preprocess}-${r.n_components}-${i}`}
                  className={`cursor-pointer ${active ? "bg-green-200" : ""}`}
                  onClick={() =>
                    setSelected({
                      preprocess: r.preprocess,
                      n_components: r.n_components,
                    })
                  }
                >
                  <td className="px-4 py-2 border-b">{prep}</td>
                  <td className="px-4 py-2 border-b">{r.n_components}</td>
                  <td className="px-4 py-2 border-b">{metric}</td>
                  <td className="px-4 py-2 border-b">{r2}</td>
                  <td className="px-4 py-2 border-b">{valMethod}</td>
                  <td className="px-4 py-2 border-b">{range}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  function renderChart(rs) {
    // agrupa por preprocess para traçar curva (n_components x métrica)
    const grouped = {};
    rs.forEach((r) => {
      const key = PREP_LABEL[r.preprocess] || r.preprocess;
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push({
        nc: r.n_components,
        val: isClass ? r?.val_metrics?.Accuracy || 0 : r?.RMSECV || 0,
      });
    });

    const traces = Object.keys(grouped).map((k) => {
      const arr = grouped[k].sort((a, b) => a.nc - b.nc);
      return {
        x: arr.map((a) => a.nc),
        y: arr.map((a) => a.val),
        mode: "lines+markers",
        name: k,
        type: "scatter",
      };
    });

    if (window?.Plotly && document.getElementById("decisionChart")) {
      window.Plotly.newPlot(
        "decisionChart",
        traces,
        {
          title: `${metricLabel} x n_components`,
          xaxis: { title: "n_components" },
          yaxis: { title: metricLabel },
        },
        { responsive: true }
      );
    }
  }

  useEffect(() => {
    if (optResults && optResults.length) {
      renderChart(optResults);
    }
  }, [optResults]); // eslint-disable-line

  async function continueWithSelection() {
    if (!selected) return;
    setError("");

    // monta o FormData para /analisar usando a seleção
    const fd = new FormData();
    fd.append("file", file);
    fd.append("target", step2.target);
    fd.append("n_components", selected.n_components);
    fd.append("classification", step2.classification ? "true" : "false");
    if (step2.threshold !== undefined) {
      fd.append("threshold", step2.threshold);
    }
    fd.append("n_bootstrap", step2.n_bootstrap || 0);
    fd.append("validation_method", step2.validation_method);
    if (step2.validation_method === "KFold") {
      fd.append(
        "validation_params",
        JSON.stringify({
          n_splits: step2?.validation_params?.n_splits || 5,
          shuffle: true,
          random_state: 42,
        })
      );
    } else if (step2.validation_method === "Holdout") {
      fd.append(
        "validation_params",
        JSON.stringify({
          test_size: step2?.validation_params?.test_size || 0.3,
        })
      );
    }
    if (result?.params?.ranges) {
      fd.append("spectral_ranges", result.params.ranges);
    }

    // só um método (ou none)
    const steps =
      selected.preprocess === "none" ? [] : [{ method: selected.preprocess }];
    if (steps.length) {
      fd.append("preprocess", JSON.stringify(steps));
    }

    try {
      const data = await postTrainForm(fd);
      const fullParams = {
        ...result?.params,
        n_components: selected.n_components,
        preprocess: selected.preprocess,
        preprocess_steps: steps,
      };
      onContinue?.(data, fullParams);
    } catch (e) {
      setError(typeof e === "string" ? e : "Erro ao executar modelagem final.");
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-[#305e6b] flex items-center">
        <i className="fas fa-balance-scale mr-2"></i>Tomada de Decisão
      </h2>

      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      {/* Botão para iniciar otimização */}
      {!optResults && !running && (
        <button
          onClick={startOptimize}
          className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center"
        >
          <i className="fas fa-cogs mr-2"></i> Otimizar Modelo
        </button>
      )}

      {/* Barra de progresso */}
      {(running || progress > 0) && (
        <>
          <div className="w-full bg-gray-200 rounded">
            <div
              className="h-2 bg-green-500 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-sm text-gray-700">{progress}%</div>
        </>
      )}

      {/* Tabela e gráfico */}
      {optResults && optResults.length > 0 && (
        <>
          {renderTable(optResults)}
          <div id="decisionChart" className="w-full h-80 mt-4" />
        </>
      )}

      <div className="flex gap-2 pt-2">
        <button className="bg-gray-200 px-3 py-2 rounded" onClick={onBack}>
          Voltar
        </button>
        <button
          className="bg-[#2e5339] hover:bg-[#305e6b] text-white px-4 py-2 rounded disabled:opacity-50"
          onClick={continueWithSelection}
          disabled={!selected}
        >
          Continuar com seleção
        </button>
      </div>
    </div>
  );
}
