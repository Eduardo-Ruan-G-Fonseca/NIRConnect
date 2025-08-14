import { useEffect, useMemo, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { postOptimize, getOptimizeStatus, postTrainForm } from "../../services/api";

/* ===== Helpers ===== */
function joinList(xs) {
  if (!xs || !xs.length) return "-";
  return xs.join(", ");
}
const nf = (v) =>
  typeof v === "number"
    ? v.toLocaleString("pt-BR", { maximumFractionDigits: 4 })
    : v;
const isPlainObject = (x) => x && typeof x === "object" && !Array.isArray(x);

const PREP_LABEL = {
  none: "Nenhum",
  snv: "SNV",
  msc: "MSC",
  sg1: "1ª Derivada",
  sg2: "2ª Derivada",
  minmax: "Min–Max",
  zscore: "Z-score",
};

/* ===== Métricas: normalização, labels e ordem ===== */
const METRIC_LABELS = {
  Accuracy: "Accuracy",
  Kappa: "Kappa",
  Precision: "Precision",
  Recall: "Recall",
  F1: "F1",
  F1_macro: "F1_macro",
  F1_micro: "F1_micro",
  MacroPrecision: "MacroPrecision",
  MacroRecall: "MacroRecall",
  MacroF1: "MacroF1",
  Sensitivity: "Sensitivity",
  Specificity: "Specificity",
  R2: "R²",
  RMSE: "RMSE",
  RMSECV: "RMSECV",
};
const ORDER_CLASS = [
  "Accuracy",
  "Kappa",
  "F1",
  "F1_macro",
  "F1_micro",
  "Precision",
  "Recall",
  "Sensitivity",
  "Specificity",
  "MacroPrecision",
  "MacroRecall",
  "MacroF1",
];
const ORDER_REGR = ["R2", "RMSE", "RMSECV"];

const fmt = (v) =>
  typeof v === "number"
    ? v.toLocaleString("pt-BR", { maximumFractionDigits: 3 })
    : String(v);

function normalizeKey(k) {
  if (!k) return k;
  // FL_macro → F1_macro ; r^2 → R2
  const key = k.replace(/^FL_/i, "F1_").replace(/f1_/i, "F1_").replace(/^R\^?2$/i, "R2");
  return key;
}
function onlyScalars(obj = {}) {
  const out = {};
  Object.entries(obj).forEach(([k, v]) => {
    if (v == null) return;
    if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") {
      out[normalizeKey(k)] = v;
    }
  });
  return out;
}
function normalizeMetrics(m) {
  if (!m) return null;
  const train = onlyScalars(m.train || m.metrics_train || {});
  const cv = onlyScalars(m.cv || m.metrics_cv || {});
  if (!Object.keys(train).length && !Object.keys(cv).length) {
    const flat = onlyScalars(m);
    return { train: flat, cv: {} };
  }
  return { train, cv };
}
function orderedEntries(obj, isClass) {
  const order = isClass ? ORDER_CLASS : ORDER_REGR;
  const keys = Object.keys(obj);
  const prioritized = keys
    .filter((k) => order.includes(k))
    .sort((a, b) => order.indexOf(a) - order.indexOf(b));
  const remaining = keys.filter((k) => !order.includes(k)).sort((a, b) => a.localeCompare(b));
  const all = [...prioritized, ...remaining];
  return all.map((k) => [METRIC_LABELS[k] || k, obj[k]]);
}

/* ===== Componente principal ===== */
export default function Step4Decision({ file, step2, result, onBack, onContinue }) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pollId, setPollId] = useState(null);
  const [optResults, setOptResults] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");

  const isClass = !!step2?.classification;
  const metricLabel = isClass ? "Accuracy" : "RMSECV";

  /* === Extrai faixa espectral [min,max] da string "min-max" === */
  const spectralRange = useMemo(() => {
    const s = result?.params?.ranges || "";
    const m = s.match(/(-?\d+(\.\d+)?)[^\d]+(-?\d+(\.\d+)?)/);
    return m ? [parseFloat(m[1]), parseFloat(m[3])] : undefined;
  }, [result]);

  /* === Lista de métodos usados no Step3 (para otimização) === */
  const preprocessingList = useMemo(() => {
    const steps = result?.params?.preprocess_steps || [];
    const methods = steps.map((p) => p.method);
    return methods.length ? methods : ["none"];
  }, [result]);

  /* === Gráficos de VIPs e Confusion Matrix (prévia) === */
  useEffect(() => {
    if (!result?.data) return;

    // VIPs
    const top = result.data.top_vips || [];
    const vipArray =
      (Array.isArray(top) && top.length
        ? top
        : Array.isArray(result.data.vip)
        ? result.data.vip
        : []) || [];

    let vipNames = [],
      vipValues = [];
    if (vipArray.length) {
      if (typeof vipArray[0] === "object" && vipArray[0] !== null) {
        vipNames = vipArray.map((it) => it.feature ?? it.name ?? it[0] ?? "");
        vipValues = vipArray.map((it) => Number(it.value ?? it[1] ?? it) || 0);
      } else if (Array.isArray(vipArray[0])) {
        vipNames = vipArray.map((it) => String(it[0]));
        vipValues = vipArray.map((it) => Number(it[1]) || 0);
      } else {
        vipNames = vipArray.map((_, i) => `Var ${i + 1}`);
        vipValues = vipArray.map(Number);
      }
    }
    if (vipValues.length) {
      Plotly.newPlot(
        "vipChart",
        [{ x: vipNames, y: vipValues, type: "bar" }],
        { margin: { t: 20, r: 10, b: 80, l: 50 }, xaxis: { automargin: true }, yaxis: { title: "VIP" } },
        { displayModeBar: false }
      );
    } else {
      const el = document.getElementById("vipChart");
      if (el) el.innerHTML = "";
    }

    // Matriz de Confusão
    const cm = result.data.metrics?.ConfusionMatrix || result.data.metrics?.confusion_matrix;
    if (Array.isArray(cm) && cm.length && Array.isArray(cm[0])) {
      Plotly.newPlot(
        "cmChart",
        [{ z: cm, type: "heatmap", colorscale: "Viridis", showscale: true }],
        { margin: { t: 20, r: 10, b: 40, l: 40 } },
        { displayModeBar: false }
      );
    } else {
      const el = document.getElementById("cmChart");
      if (el) el.innerHTML = "";
    }
  }, [result?.data]);

  /* === Limpa polling ao desmontar === */
  useEffect(() => {
    return () => {
      if (pollId) clearInterval(pollId);
    };
  }, [pollId]);

  /* === Otimização === */
  async function startOptimize() {
    setError("");
    setRunning(true);
    setProgress(0);
    setOptResults(null);
    setSelected(null);

    // polling do /optimize/status
    const id = setInterval(async () => {
      try {
        const s = await getOptimizeStatus();
        const pct = s.total ? Math.round((s.current / s.total) * 100) : 0;
        setProgress(pct);
      } catch {
        /* ignora erros de polling */
      }
    }, 1000);
    setPollId(id);

    try {
      const payload = {
        target: step2.target,
        validation_method: step2.validation_method,
        n_components: parseInt(step2.n_components, 10),
        n_bootstrap: parseInt(step2.n_bootstrap || 0, 10),
        folds: step2.validation_method === "KFold" ? step2.validation_params?.n_splits || 5 : undefined,
        analysis_mode: step2.classification ? "PLS-DA" : "PLS-R",
        spectral_range: spectralRange || undefined, // fallback: back usa faixa total
        preprocessing_methods: preprocessingList?.length ? preprocessingList : ["none"],
      };

      const data = await postOptimize(file, payload);
      setOptResults(data.results || []);
      setProgress(100);
    } catch (e) {
      setError(typeof e === "string" ? e : e?.message || "Falha na otimização.");
    } finally {
      setRunning(false);
      if (pollId) clearInterval(pollId);
      setPollId(null);
    }
  }

  function sortResults(rs) {
    const arr = [...rs];
    return arr.sort((a, b) => {
      const aval = isClass ? -(a?.val_metrics?.Accuracy || 0) : a?.RMSECV ?? 1e9;
      const bval = isClass ? -(b?.val_metrics?.Accuracy || 0) : b?.RMSECV ?? 1e9;
      return aval - bval;
    });
  }

  function renderTable(rs) {
    const sorted = sortResults(rs);
    return (
      <div className="overflow-auto">
        <table className="min-w-full text-sm table-fixed">
          <colgroup>
            <col style={{ width: "24%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "18%" }} />
            <col style={{ width: "18%" }} />
          </colgroup>
          <thead>
            <tr>
              <th className="px-4 py-2 border-b text-left">Pré-processamento</th>
              <th className="px-4 py-2 border-b text-right">n_comp.</th>
              <th className="px-4 py-2 border-b text-right">{metricLabel}</th>
              <th className="px-4 py-2 border-b text-right">R²</th>
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
                r?.val_metrics?.R2 !== undefined ? Number(r.val_metrics.R2).toFixed(3) : "-";
              const valMethod = r?.validation?.method || "-";
              const range =
                r?.wl_used?.length ? `${r.wl_used[0]}-${r.wl_used[r.wl_used.length - 1]}` : "-";
              const active =
                selected && selected.preprocess === r.preprocess && selected.n_components === r.n_components;

              return (
                <tr
                  key={`${r.preprocess}-${r.n_components}-${i}`}
                  className={`cursor-pointer ${active ? "bg-green-200" : ""}`}
                  onClick={() => setSelected({ preprocess: r.preprocess, n_components: r.n_components })}
                >
                  <td className="px-4 py-2 border-b truncate" title={prep}>
                    {prep}
                  </td>
                  <td className="px-4 py-2 border-b text-right">{r.n_components}</td>
                  <td className="px-4 py-2 border-b text-right">{metric}</td>
                  <td className="px-4 py-2 border-b text-right">{r2}</td>
                  <td className="px-4 py-2 border-b">{valMethod}</td>
                  <td className="px-4 py-2 border-b break-words">{range}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  /* === Chart da decisão (curvas) após otimização === */
  useEffect(() => {
    if (!optResults || !optResults.length) return;

    const grouped = {};
    optResults.forEach((r) => {
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

    Plotly.newPlot(
      "decisionChart",
      traces,
      { title: `${metricLabel} x n_components`, xaxis: { title: "n_components" }, yaxis: { title: metricLabel } },
      { responsive: true }
    );
  }, [optResults]); // eslint-disable-line

  /* === Continuar com a seleção escolhida na tabela === */
  async function continueWithSelection() {
    if (!selected) return;
    setError("");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("target", step2.target);
    fd.append("n_components", selected.n_components);
    fd.append("classification", step2.classification ? "true" : "false");
    if (step2.threshold !== undefined) fd.append("threshold", step2.threshold);
    fd.append("n_bootstrap", step2.n_bootstrap || 0);
    fd.append("validation_method", step2.validation_method);

    if (step2.validation_method === "KFold") {
      fd.append(
        "validation_params",
        JSON.stringify({ n_splits: step2?.validation_params?.n_splits || 5, shuffle: true, random_state: 42 })
      );
    } else if (step2.validation_method === "Holdout") {
      fd.append("validation_params", JSON.stringify({ test_size: step2?.validation_params?.test_size || 0.3 }));
    }
    if (result?.params?.ranges) fd.append("spectral_ranges", result.params.ranges);

    const steps = selected.preprocess === "none" ? [] : [{ method: selected.preprocess }];
    if (steps.length) fd.append("preprocess", JSON.stringify(steps));

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

  /* === Render === */
  const faixaStr = result?.params?.ranges || result?.data?.range_used || "-";
  const preprocessStr = joinList(result?.params?.preprocess_steps?.map((p) => p.method));

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-[#305e6b] flex items-center">
        <i className="fas fa-balance-scale mr-2"></i>Tomada de Decisão
      </h2>

      {error && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">{error}</div>}

      {/* ===== PRÉVIA DO MODELO ===== */}
      {result?.data && (
        <section className="grid gap-6 grid-cols-1 md:grid-cols-12 items-start">
          {/* Parâmetros */}
          <div className="nir-card md:col-span-4">
            <h4 className="nir-card-title">Parâmetros</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="nir-kv">
                <div>Alvo</div>
                <b className="break-words">{step2?.target || "-"}</b>
                <div>Modo</div>
                <b>{step2?.classification ? "PLS-DA" : "PLS-R"}</b>
                <div>Validação</div>
                <b>{step2?.validation_method || "-"}</b>
              </div>
              <div className="nir-kv">
                <div>Componentes (PLS)</div>
                <b>{step2?.n_components ?? "-"}</b>
                <div>Faixa espectral</div>
                <b className="break-words">{faixaStr}</b>
              </div>
              <div className="nir-kv">
                <div>Pré‑processos</div>
                <b className="break-words">{preprocessStr}</b>
              </div>
            </div>
          </div>

          {/* Métricas */}
          {result?.data?.metrics && (
            <div className="nir-card md:col-span-8">
              <h4 className="nir-card-title">Métricas</h4>
              <div className="max-h-[360px] overflow-auto pr-1">
                {(() => {
                  const norm = normalizeMetrics(result.data.metrics);
                  if (!norm) return <div className="nir-muted">Sem métricas.</div>;

                  const hasTrain = Object.keys(norm.train).length > 0;
                  const hasCV = Object.keys(norm.cv).length > 0;

                  const rowsTrain = orderedEntries(norm.train, isClass);
                  const rowsCV = orderedEntries(norm.cv, isClass);

                  const Table = ({ rows }) => (
                    <div className="nir-table-wrap">
                      <table className="nir-table">
                        <colgroup>
                          <col className="nir-col-k" />
                          <col className="nir-col-v" />
                        </colgroup>
                        <thead>
                          <tr>
                            <th className="text-left sticky top-0 bg-[#fafafa]">Métrica</th>
                            <th className="text-right sticky top-0 bg-[#fafafa]">Valor</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map(([k, v]) => (
                            <tr key={k}>
                              <td className="nir-col-k">{k}</td>
                              <td className="nir-col-v">{fmt(v)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  );

                  if (hasTrain || hasCV) {
                    return (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {hasTrain && (
                          <div className="nir-subcard">
                            <div className="nir-subcard-title">Treino</div>
                            <Table rows={rowsTrain} />
                          </div>
                        )}
                        {hasCV && (
                          <div className="nir-subcard">
                            <div className="nir-subcard-title">Validação</div>
                            <Table rows={rowsCV} />
                          </div>
                        )}
                      </div>
                    );
                  }
                  return <Table rows={rowsTrain} />;
                })()}
              </div>
            </div>
          )}

          {/* VIPs */}
          <div className="nir-card md:col-span-6">
            <h4 className="nir-card-title">VIPs (Top)</h4>
            <div id="vipChart" className="nir-chart h-64" />
            {!((result?.data?.top_vips && result.data.top_vips.length) || (result?.data?.vip && result.data.vip.length)) && (
              <div className="nir-muted">Sem VIPs disponíveis para esta calibração.</div>
            )}
          </div>

          {/* Matriz de Confusão */}
          <div className="nir-card md:col-span-6">
            <h4 className="nir-card-title">Matriz de Confusão</h4>
            <div id="cmChart" className="nir-chart h-64" />
            {!(result?.data?.metrics?.ConfusionMatrix || result?.data?.metrics?.confusion_matrix) && (
              <div className="nir-muted">Não disponível para este modo/validação.</div>
            )}
          </div>
        </section>
      )}

      {/* ===== Ações de Otimização ===== */}
      {!optResults && !running && (
        <button onClick={startOptimize} className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center">
          <i className="fas fa-cogs mr-2"></i> Otimizar Modelo
        </button>
      )}

      {(running || progress > 0) && (
        <>
          <div className="w-full bg-gray-200 rounded">
            <div className="h-2 bg-green-500 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <div className="text-sm text-gray-700">{progress}%</div>
        </>
      )}

      {optResults &&
        (optResults.length > 0 ? (
          <>
            {renderTable(optResults)}
            <div id="decisionChart" className="w-full h-80 mt-4" />
          </>
        ) : (
          <div className="text-sm text-gray-700">
            Nenhum resultado foi retornado pela otimização. Tente reduzir o número de componentes ou trocar o método de
            validação (ex.: KFold).
          </div>
        ))}

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
