import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { getOptimizeStatus } from "../../services/api";
import { postJSON } from "../../lib/api";

/* ===== Helpers ===== */
function joinList(xs){ if(!xs || !xs.length) return "-"; return xs.join(", "); }
const fmt = (v) => (typeof v === "number" ? v.toLocaleString("pt-BR", { maximumFractionDigits: 3 }) : String(v));

/* ===== Labels ===== */
const PREP_LABEL = {
  none:"Nenhum", snv:"SNV", msc:"MSC",
  sg1:"1ª Derivada", sg2:"2ª Derivada",
  minmax:"Min–Max", zscore:"Z-score",
};

/* ===== Normalização/ordem de MÉTRICAS ===== */
const CANON = {
  accuracy:"Accuracy", kappa:"Kappa", precision:"Precision", recall:"Recall", f1:"F1",
  "fl_macro":"F1_macro","f1_macro":"F1_macro","f1-macro":"F1_macro","macro_f1":"F1_macro","f1macro":"F1_macro",
  "fl_micro":"F1_micro","f1_micro":"F1_micro","f1-micro":"F1_micro","microf1":"F1_micro",
  "macroprecision":"MacroPrecision","macro_precision":"MacroPrecision",
  "macrorecall":"MacroRecall","macro_recall":"MacroRecall",
  "macrof1":"MacroF1","macro_f1score":"MacroF1",
  sensitivity:"Sensitivity", specificity:"Specificity",
  r2:"R2","r^2":"R2","r²":"R2",
  rmse:"RMSE",
  rmsecv:"RMSECV","rmse_cv":"RMSECV","rmse-cv":"RMSECV",
};
const LABEL = {
  Accuracy:"Accuracy", Kappa:"Kappa", Precision:"Precision", Recall:"Recall",
  F1:"F1", F1_macro:"F1_macro", F1_micro:"F1_micro",
  MacroPrecision:"MacroPrecision", MacroRecall:"MacroRecall", MacroF1:"MacroF1",
  Sensitivity:"Sensitivity", Specificity:"Specificity",
  R2:"R²", RMSE:"RMSE", RMSECV:"RMSECV",
};
const ORDER_CLASS = [
  "Accuracy","Kappa","F1","F1_macro","F1_micro",
  "Precision","Recall","Sensitivity","Specificity",
  "MacroPrecision","MacroRecall","MacroF1",
];
const ORDER_REGR = ["R2","RMSE","RMSECV"];

const isPlainObject = (x) => x && typeof x === "object" && !Array.isArray(x);
function canonKey(k){
  if(!k) return k;
  const key = String(k).trim(); const lk = key.toLowerCase();
  if(/^r\^?2$|^r2$|^r²$/.test(lk)) return "R2";
  if(lk.startsWith("fl_")) return "F1_" + key.split("_")[1];
  return CANON[lk] || key;
}
function onlyScalars(obj = {}) {
  const out = {};
  Object.entries(obj).forEach(([k, v]) => {
    if (v == null) return;

    // pula textos descritivos longos (ex.: SensitivityDescription)
    if (/description$/i.test(String(k))) return;

    if (["number", "string", "boolean"].includes(typeof v)) {
      out[canonKey(k)] = v;
    }
  });
  return out;
}

function normalizeMetrics(m){
  if(!m) return null;
  const trainRaw = m.train || m.training || m.metrics_train || m.Treino || {};
  const cvRaw    = m.cv || m.validation || m.metrics_cv || m.Validação || m.val || {};
  const train = isPlainObject(trainRaw) ? onlyScalars(trainRaw) : {};
  const cv    = isPlainObject(cvRaw)    ? onlyScalars(cvRaw)    : {};
  if(!Object.keys(train).length && !Object.keys(cv).length){
    const flat = onlyScalars(m);
    return { train: flat, cv: {} };
  }
  return { train, cv };
}
function orderedEntries(obj, isClass){
  const order = isClass ? ORDER_CLASS : ORDER_REGR;
  const keys = Object.keys(obj);
  const prior = keys.filter(k=>order.includes(k)).sort((a,b)=>order.indexOf(a)-order.indexOf(b));
  const rest  = keys.filter(k=>!order.includes(k)).sort((a,b)=>a.localeCompare(b));
  return [...prior, ...rest].map(k => [LABEL[k] || k, obj[k]]);
}

export default function Step4Decision({ step2, result, onBack, onContinue }) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const pollRef = useRef(null);
  const [optData, setOptData] = useState(null);
  const [optResults, setOptResults] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [busy, setBusy] = useState(false);

  // Refs para gráficos Plotly (evita IDs globais)
  const vipRef = useRef(null);
  const cmRef = useRef(null);
  const decisionRef = useRef(null);

  const isClass = !!step2?.classification;
  const metricLabel = isClass ? "Accuracy" : "RMSECV";

  const spectralRange = useMemo(() => {
    const s = result?.params?.ranges || "";
    const m = s.match(/(-?\d+(\.\d+)?)[^\d]+(-?\d+(\.\d+)?)/);
    return m ? [parseFloat(m[1]), parseFloat(m[3])] : undefined;
  }, [result]);


  /* ===== Gráficos: VIPs e Confusion ===== */
  useEffect(() => {
    if (!result?.data) return;

    // VIPs
    const top = result.data.top_vips || [];
    const vipArray = (Array.isArray(top) && top.length ? top : Array.isArray(result.data.vip) ? result.data.vip : []) || [];
    let vipNames = [], vipValues = [];
    if (vipArray.length) {
      if (typeof vipArray[0] === "object" && vipArray[0] !== null) {
        vipNames  = vipArray.map(it => it.feature ?? it.name ?? it[0] ?? "");
        vipValues = vipArray.map(it => {
          const v = Number(it.value ?? it[1] ?? it);
          return Number.isFinite(v) ? v : 0;
        });
      } else if (Array.isArray(vipArray[0])) {
        vipNames  = vipArray.map(it => String(it[0]));
        vipValues = vipArray.map(it => {
          const v = Number(it[1]);
          return Number.isFinite(v) ? v : 0;
        });
      } else {
        vipNames  = vipArray.map((_, i) => `Var ${i+1}`);
        vipValues = vipArray.map(v => {
          const n = Number(v);
          return Number.isFinite(n) ? n : 0;
        });
      }
    }
    if (vipValues.length && vipRef.current) {
      Plotly.newPlot(
        vipRef.current,
        [{ x: vipNames, y: vipValues, type: "bar" }],
        { margin: { t: 20, r: 10, b: 80, l: 50 }, xaxis: { automargin: true }, yaxis: { title: "VIP" } },
        { displayModeBar: false }
      );
    } else if (vipRef.current) {
      Plotly.purge(vipRef.current);
      vipRef.current.innerHTML = "";
    }

    // Confusion matrix
    const cm = result.data.metrics?.ConfusionMatrix || result.data.metrics?.confusion_matrix;
    if (Array.isArray(cm) && cm.length && Array.isArray(cm[0]) && cmRef.current) {
      Plotly.newPlot(
        cmRef.current,
        [{ z: cm, type: "heatmap", colorscale: "Viridis", showscale: true }],
        { margin: { t: 20, r: 10, b: 40, l: 40 } },
        { displayModeBar: false }
      );
    } else if (cmRef.current) {
      Plotly.purge(cmRef.current);
      cmRef.current.innerHTML = "";
    }

    // resize on window change
    function handleResize(){
      if (vipRef.current) Plotly.Plots.resize(vipRef.current);
      if (cmRef.current) Plotly.Plots.resize(cmRef.current);
    }
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [result?.data]);

  useEffect(() => {
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, []);

  /* ===== Otimização ===== */
  const POLL_INTERVAL_MS = 1000;
  async function handleOptimize() {
    setError("");
    setBusy(true);
    setRunning(true);
    setProgress(0);
    setOptResults(null);
    setOptData(null);
    setSelected(null);

    // polling progress
    pollRef.current = setInterval(async () => {
      try {
        const s = await getOptimizeStatus();
        const current = Number(s?.current ?? 0);
        const total = Number(s?.total ?? 0);
        const pct = total > 0 ? Math.round((current / total) * 100) : 0;
        setProgress(Math.max(0, Math.min(100, pct)));
        if (total > 0 && current >= total) {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        /* ignora polling error */
      }
    }, POLL_INTERVAL_MS);

    try {
      const rangeStr = result?.params?.ranges || (spectralRange ? `${spectralRange[0]}-${spectralRange[1]}` : undefined);
      const payload = {
        target: step2.target,
        n_components: step2.n_components,
        classification: isClass,
        threshold: step2.threshold,
        n_bootstrap: 0,
        validation_method: step2.validation_method,
        validation_params: step2.validation_params,
        spectral_ranges: rangeStr
      };
      const res = await postJSON("/optimize", payload);
      setOptData(res);
      const arr = sortResults(res?.results || []);
      setOptResults(arr);
      if (arr.length === 0) {
        setError("Nenhuma combinação válida encontrada. Revise n_components, pré-processos ou validação.");
      } else {
        if (res?.best?.preprocess) {
          const idx = arr.findIndex(
            r => r.preprocess === res.best.preprocess && r.n_components === res.best.n_components
          );
          setSelected(idx >= 0 ? idx : 0);
        } else {
          setSelected(0);
        }
      }
      setRunning(false);
      setProgress(100);
      if (pollRef.current) clearInterval(pollRef.current);
    } catch (e) {
      setError(typeof e === "string" ? e : e?.message || "Falha na otimização.");
    } finally {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      setRunning(false);
      setBusy(false);
    }
  }

  function sortResults(rs){
    const arr = [...rs];
    return arr.sort((a,b)=>{
      const aval = isClass
        ? -((a?.Accuracy ?? a?.val_metrics?.Accuracy) || 0)
        : ((a?.RMSECV ?? a?.val_metrics?.RMSECV) || Number.POSITIVE_INFINITY);
      const bval = isClass
        ? -((b?.Accuracy ?? b?.val_metrics?.Accuracy) || 0)
        : ((b?.RMSECV ?? b?.val_metrics?.RMSECV) || Number.POSITIVE_INFINITY);
      return aval - bval;
    });
  }
  function renderTable(rs){
    return (
      <div className="nir-table-wrap">
        <table className="nir-table full">
          <colgroup>
            <col style={{width:"28%"}} />
            <col style={{width:"12%"}} />
            <col style={{width:"14%"}} />
            <col style={{width:"14%"}} />
            <col style={{width:"16%"}} />
            <col style={{width:"16%"}} />
          </colgroup>
          <thead>
            <tr>
              <th>Pré-processamento</th>
              <th className="text-right">n_comp.</th>
              <th className="text-right">{metricLabel}</th>
              <th className="text-right">{isClass ? "MacroF1" : "R2CV"}</th>
              <th>Validação</th>
              <th>Faixa usada</th>
            </tr>
          </thead>
          <tbody>
            {rs.map((r,i)=>{
              const prepRaw = r.preprocess ?? r.prep;
              const prep = PREP_LABEL[prepRaw] || prepRaw || "-";
              const val = isClass
                ? ((r?.Accuracy ?? r?.val_metrics?.Accuracy) || 0)
                : ((r?.RMSECV ?? r?.val_metrics?.RMSECV) || 0);
              const metric = Number(val).toFixed(3);
              const secRaw = isClass ? (r?.MacroF1 ?? r?.val_metrics?.MacroF1) : (r?.R2CV ?? r?.val_metrics?.R2CV);
              const sec = secRaw !== undefined && Number.isFinite(Number(secRaw)) ? Number(secRaw).toFixed(3) : "-";
              const valMethod = r?.validation?.method || r?.validation || "-";
              const range = r?.range || (r?.wl_used?.length ? `${r.wl_used[0]}-${r.wl_used[r.wl_used.length-1]}` : "-");
              const active = selected === i;
              return (
                <tr
                  key={`${r.id || `${prepRaw ?? "none"}-${r.n_components ?? "nc"}`}`}
                  className={active ? "row-active" : ""}
                  onClick={()=>setSelected(i)}
                  style={{ cursor: "pointer" }}
                >
                  <td>{prep}</td>
                  <td className="text-right tabular-nums">{r.n_components}</td>
                  <td className="text-right tabular-nums font-semibold">{metric}</td>
                  <td className="text-right tabular-nums">{sec}</td>
                  <td>{valMethod}</td>
                  <td>{range}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  /* ===== Curvas ===== */
  useEffect(() => {
    if (!optData?.curves || !decisionRef.current) return;
    const traces = (optData.curves || []).map(c => {
      const xs = c.points.map(p => p.n_components);
      const ys = c.points.map(p => (isClass ? p.MacroF1 : p.RMSECV));
      return { x: xs, y: ys, mode: "lines+markers", name: c.preprocess, type: "scatter" };
    });
    Plotly.newPlot(
      decisionRef.current,
      traces,
      {
        title: "Variáveis Latentes × Métrica",
        xaxis: { title: "VL (n_components)" },
        yaxis: { title: isClass ? "MacroF1 / Acurácia" : "RMSECV" },
        legend: { orientation: "h" }
      },
      { responsive: true, displaylogo: false }
    );

    function handleResize() {
      if (decisionRef.current) Plotly.Plots.resize(decisionRef.current);
    }
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      if (decisionRef.current) Plotly.purge(decisionRef.current);
    };
  }, [optData, isClass]);

  /* ===== Continuar ===== */
  async function handleTrainWithSelection(){
    if(selected == null || !optData) return; setError("");
    const choice = optResults?.[selected];
    if (!choice) return;
    setBusy(true);
    setSaving(true);
    try {
      const rangeStr = result?.params?.ranges || (optData.range_used ? `${optData.range_used[0]}-${optData.range_used[1]}` : undefined);
      await postJSON("/train", {
        target: step2.target,
        n_components: choice.n_components,
        classification: isClass,
        preprocess: choice.preprocess,
        validation_method: step2.validation_method,
        validation_params: step2.validation_params,
        spectral_ranges: rangeStr
      });
      onContinue?.(optData, { ...result?.params, n_components: choice.n_components, preprocess: choice.preprocess });
    } catch (e) {
      setError(typeof e === "string" ? e : (e?.message || "Erro ao executar modelagem final."));
    } finally {
      setSaving(false);
      setBusy(false);
    }
  }

  const faixaStr = result?.params?.ranges || result?.data?.range_used || "-";
// nomes amigáveis (ex.: sg1 -> "1ª Derivada")
  const preprocessStr = joinList(
    result?.params?.preprocess_steps?.map(p => PREP_LABEL[p.method] || p.method)
  );


  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold text-[#305e6b] flex items-center">
        <i className="fas fa-balance-scale mr-2"></i>Tomada de Decisão
      </h2>

      {error && <div className="text-base text-red-700 bg-red-50 border border-red-200 rounded p-3">{error}</div>}

      {result?.data && (
        <div className="grid gap-6 grid-cols-1 md:grid-cols-12 items-start nir-preview">
          {/* Parâmetros */}
          <div className="nir-card md:col-span-5 xl:col-span-4 2xl:col-span-4">
            <h4 className="nir-card-title">Parâmetros</h4>

            <div className="nir-kv nir-kv--wide nir-params">
              <div>Alvo</div>                 <b>{step2?.target || "-"}</b>
              <div>Modo</div>                 <b>{step2?.classification ? "PLS-DA" : "PLS-R"}</b>
              <div>Validação</div>            <b>{step2?.validation_method || "-"}</b>
              <div>Componentes (PLS)</div>    <b>{step2?.n_components ?? "-"}</b>
              <div>Faixa espectral</div>      <b>{faixaStr}</b>
              <div>Pré‑processos</div>        <b>{preprocessStr || "Nenhum"}</b>
            </div>
          </div>



          {/* Métricas */}
          {result?.data?.metrics && (
            <div className="nir-card md:col-span-7 xl:col-span-8 2xl:col-span-8">
              <h4 className="nir-card-title">Métricas</h4>
              {(() => {
                const norm = normalizeMetrics(result.data.metrics);
                if (!norm) return <div className="nir-muted">Sem métricas.</div>;

                const hasTrain = Object.keys(norm.train).length > 0;
                const hasCV    = Object.keys(norm.cv).length > 0;
                const rowsTrain = orderedEntries(norm.train, isClass);
                const rowsCV    = orderedEntries(norm.cv,    isClass);

                const Table = ({ rows }) => (
                <div className="nir-table-wrap">
                  <table className="nir-table compact full">
                    <colgroup>
                      <col style={{ width: "50%" }} />
                      <col style={{ width: "50%" }} />
                    </colgroup>
                    <thead>
                      <tr>
                        <th>Métrica</th>
                        <th className="text-right">Valor</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map(([k, v]) => (
                        <tr key={k}>
                          <td>{k}</td>
                          <td className="text-right font-semibold tabular-nums">{fmt(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );

              // …
              if (hasTrain || hasCV) {
                return (
                  <div className="metrics-2col">
                    {hasTrain && (
                      <div className="metrics-card">
                        <div className="nir-subcard-title">Treino</div>
                        <Table rows={rowsTrain} />
                      </div>
                    )}
                    {hasCV && (
                      <div className="metrics-card">
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
          )}

          {/* VIPs */}
          <div className="nir-card md:col-span-6">
            <h4 className="nir-card-title">VIPs (Top)</h4>
            {((result?.data?.top_vips && result.data.top_vips.length) || (result?.data?.vip && result.data.vip.length)) ? (
              <div ref={vipRef} className="nir-chart h-80" />
            ) : (
              <div className="nir-empty">Sem VIPs disponíveis para esta calibração.</div>
            )}
          </div>

          {/* Matriz de Confusão */}
          <div className="nir-card md:col-span-6">
            <h4 className="nir-card-title">Matriz de Confusão</h4>
            {(result?.data?.metrics?.ConfusionMatrix || result?.data?.metrics?.confusion_matrix) ? (
              <div ref={cmRef} className="nir-chart h-80" />
            ) : (
              <div className="nir-empty">Não disponível para este modo/validação.</div>
            )}
          </div>

        </div>
      )}

      {/* Ações de Otimização */}
      {!optResults && (
        <button
          onClick={handleOptimize}
          className="bg-blue-600 hover:bg-blue-700 text-white py-3 px-5 rounded-lg shadow-sm flex items-center disabled:opacity-50"
          disabled={busy}
        >
          {busy ? <i className="fas fa-spinner fa-spin mr-2"></i> : <i className="fas fa-cogs mr-2"></i>}
          Otimizar Modelo
        </button>
      )}

      {(running || progress > 0) && (
        <>
          {step2?.validation_method === "LOO" && (
            <div className="text-xs text-amber-700 mb-2">
              Executando Leave-One-Out — cada combinação roda N vezes (N = número de amostras). Pode levar mais tempo.
            </div>
          )}
          <div className="w-full bg-gray-200 rounded">
            <div className="h-2 bg-green-500 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <div className="text-sm text-gray-700">{progress}%</div>
        </>
      )}

      {optResults && (
        optResults.length > 0 ? (
          <>
            {optData && (
              <div className="nir-card mb-4">
                <div className="nir-kv nir-params">
                  <div>Validação</div> <b>{optData.validation_used}</b>
                  {optData.range_used && (
                    <>
                      <div>Faixa usada</div>
                      <b>{optData.range_used[0]}–{optData.range_used[1]} nm</b>
                    </>
                  )}
                </div>
                {optData.best?.val_metrics && (
                  <div className="nir-table-wrap mt-2">
                    <table className="nir-table compact full">
                      <colgroup>
                        <col style={{ width: "50%" }} />
                        <col style={{ width: "50%" }} />
                      </colgroup>
                      <thead>
                        <tr><th>Métrica</th><th className="text-right">Valor</th></tr>
                      </thead>
                      <tbody>
                        {isClass ? (
                          <>
                            <tr><td>Accuracy</td><td className="text-right">{fmt(optData.best.val_metrics.Accuracy)}</td></tr>
                            <tr><td>MacroF1</td><td className="text-right">{fmt(optData.best.val_metrics.MacroF1)}</td></tr>
                          </>
                        ) : (
                          <>
                            <tr><td>RMSECV</td><td className="text-right">{fmt(optData.best.val_metrics.RMSECV)}</td></tr>
                            <tr><td>R2CV</td><td className="text-right">{fmt(optData.best.val_metrics.R2CV)}</td></tr>
                          </>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
                {isClass && optData.best?.val_metrics?.per_class && (
                  <div className="nir-table-wrap mt-4">
                    <table className="nir-table compact full">
                      <thead>
                        <tr>
                          <th>Classe</th><th className="text-right">Precisão</th><th className="text-right">Revocação</th><th className="text-right">F1</th><th className="text-right">Suporte</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(optData.labels_all || Object.keys(optData.best.val_metrics.per_class)).map(lbl => {
                          const d = optData.best.val_metrics.per_class[String(lbl)] || {};
                          return (
                            <tr key={lbl}>
                              <td>{lbl}</td>
                              <td className="text-right">{fmt(d.precision)}</td>
                              <td className="text-right">{fmt(d.recall)}</td>
                              <td className="text-right">{fmt(d.f1)}</td>
                              <td className="text-right">{fmt(d.support)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
            <div className="nir-card">{renderTable(optResults)}</div>
            <div ref={decisionRef} className="w-full h-96 mt-6" />
          </>
        ) : (
          <div className="text-base text-gray-700">
            Nenhum resultado foi retornado pela otimização.
            Tente reduzir o número de componentes ou trocar o método de validação (ex.: KFold).
          </div>
        )
      )}

      <div className="flex gap-3 pt-2">
        <button className="bg-gray-200 px-4 py-2 rounded-lg" onClick={onBack}>Voltar</button>
        <button
          className="bg-[#2e5339] hover:bg-[#305e6b] text-white px-5 py-2 rounded-lg disabled:opacity-50 flex items-center"
          onClick={handleTrainWithSelection}
          disabled={busy || saving || selected == null || !optResults || optResults.length === 0}
        >
          {saving && <i className="fas fa-spinner fa-spin mr-2"></i>}
          Continuar com seleção
        </button>
      </div>
    </div>
  );
}
