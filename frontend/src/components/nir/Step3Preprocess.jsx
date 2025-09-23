// src/components/nir/Step3Preprocess.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

import { postTrain } from "../../services/api";
import { getDatasetId } from "../../api/http";

// util local
const parseWavelengths = (meta) => {
  // prioridade: meta.spectra_matrix.wavelengths
  if (meta?.spectra_matrix?.wavelengths && meta.spectra_matrix.wavelengths.length > 0) {
    return meta.spectra_matrix.wavelengths.map(Number);
  }
  // fallback: meta.columns (strings tipo "908,1")
  return (meta?.columns || []).map(c => {
    const s = String(c).trim().replace(',', '.');
    const v = Number(s);
    return Number.isFinite(v) ? v : null;
  });
};

export default function Step3Preprocess({ meta, step2, onBack, onAnalyzed }) {
  const chartRef = useRef(null);
  const preselectedOnce = useRef(false); // garante que a seleção total só acontece 1x

  // parse metadados e matriz
  const { wavelengths, series } = useMemo(() => {
    let wl = parseWavelengths(meta);
    let M = meta?.spectra_matrix?.values || [];

    // Se wavelengths vier com algum null, gera eixo 0..n_wavelengths-1 para não quebrar o grafico
    if (!wl || wl.length !== (M[0]?.length || 0) || wl.some(v => v === null)) {
      wl = Array.from({ length: M[0]?.length || 0 }, (_, i) => i);
    }

    return { wavelengths: wl, series: M };
  }, [meta]);

  // ATENÇÃO: series já está corrigida na orientação acima
  const meanLine = useMemo(() => {
    if (meta.mean_spectra?.values && meta.mean_spectra.values.length === wavelengths.length) {
      return meta.mean_spectra.values;
    }
    if (!series?.length) return [];
    const nS = series.length, nW = series[0].length;
    const acc = new Array(nW).fill(0);
    for (let i = 0; i < nS; i++) for (let j = 0; j < nW; j++) acc[j] += Number(series[i][j] ?? 0);
    return acc.map(v => v / nS);
  }, [meta, wavelengths, series]);

  const wlMinAuto = wavelengths[0] ?? "";
  const wlMaxAuto = wavelengths.length ? wavelengths[wavelengths.length - 1] : "";

  const [lambdaMin, setLambdaMin] = useState(wlMinAuto);
  const [lambdaMax, setLambdaMax] = useState(wlMaxAuto);
  const [showMean, setShowMean]   = useState(true);
  const [ranges, setRanges]       = useState([]); // [[min,max]]
  const [running, setRunning]     = useState(false);

  // ao carregar o dataset, preenche min/max e pré-seleciona TODA a faixa uma única vez
  useEffect(() => {
    if (!wavelengths.length) return;
    setLambdaMin(wavelengths[0]);
    setLambdaMax(wavelengths[wavelengths.length - 1]);

    if (!preselectedOnce.current) {
      setRanges([[wavelengths[0], wavelengths[wavelengths.length - 1]]]);
      preselectedOnce.current = true;
    }
  }, [wavelengths]);

  const toNum = (v) => Number(String(v).replace(",", "."));

  // gráfico
  useEffect(() => {
    if (!chartRef.current || !wavelengths.length) return;

    const traces = [];
    const MAX_LINES = 300;
    const toPlot = series.slice(0, MAX_LINES);
    if (toPlot.length) {
      for (const row of toPlot) {
        traces.push({
          x: wavelengths,
          y: row,
          type: "scatter",
          mode: "lines",
          line: { width: 1, color: "#999" },
          opacity: 0.15,
          hoverinfo: "skip",
          name: "Espectros",
        });
      }
    }

    if (showMean && meanLine.length) {
      traces.push({
        x: wavelengths,
        y: meanLine,
        type: "scatter",
        mode: "lines",
        line: { width: 2, color: "red" },
        name: "Médio",
      });
    }

    const shapes = ranges.map(([a, b]) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: Math.min(a, b),
      x1: Math.max(a, b),
      y0: 0,
      y1: 1,
      fillcolor: "rgba(16,185,129,0.2)", // verde
      line: { width: 0 },
    }));

    const flat = series.flat();
    const ymin = Math.min(...flat);
    const ymax = Math.max(...flat);
    const pad = (ymax - ymin) * 0.05;
    const layout = {
      margin: { l: 40, r: 10, t: 10, b: 40 },
      dragmode: "select",
      shapes,
      yaxis: { range: [ymin - pad, ymax + pad] },
    };

    const node = chartRef.current;
    if (!node) return () => {};

    Plotly.react(node, traces, layout, { responsive: true, displayModeBar: false });

    const onSelected = (ev) => {
      if (ev?.range?.x) {
        const [a, b] = ev.range.x;
        const aR = Math.round(a);
        const bR = Math.round(b);
        setLambdaMin(aR);
        setLambdaMax(bR);
        setRanges([[Math.min(aR, bR), Math.max(aR, bR)]]);
      }
      Plotly.relayout(node, { dragmode: "select" });
    };

    node.on("plotly_selected", onSelected);

    return () => {
      node?.removeAllListeners?.("plotly_selected");
      try { Plotly.purge(node); } catch { /* ignore */ }
    };
  }, [wavelengths, series, meanLine, showMean, ranges]);

  function applyRange() {
    const a = toNum(lambdaMin), b = toNum(lambdaMax);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return;
    setRanges([[Math.min(a, b), Math.max(a, b)]]);
  }

  function selectAll() {
    if (!wavelengths.length) return;
    setLambdaMin(wlMinAuto);
    setLambdaMax(wlMaxAuto);
    setRanges([[wlMinAuto, wlMaxAuto]]);
  }

  function clearRanges() {
    setRanges([]);
  }

  const selectedText = ranges.length
    ? ranges.map(([a,b]) => `${a}–${b} nm`).join(", ")
    : "—";

  async function runAnalysis(e) {
    e.preventDefault();
    if (!ranges.length) return alert("Selecione ao menos uma faixa.");

    setRunning(true);
    try {
      const checked = Array.from(
        document.querySelectorAll('input[name="pp-method"]:checked')
      ).map((i) => i.value);

      const rangesStr = ranges.map(([a, b]) => `${a}-${b}`).join(",");
      const firstValidRange = ranges.find(([a, b]) =>
        Number.isFinite(a) && Number.isFinite(b) && a !== b
      );
      const spectralRange = firstValidRange
        ? {
            min: Math.min(firstValidRange[0], firstValidRange[1]),
            max: Math.max(firstValidRange[0], firstValidRange[1]),
          }
        : null;

      const dsId = getDatasetId();
      if (!dsId) throw new Error("Dataset não encontrado. Volte ao passo 1 e reenvie o arquivo.");
      const validationMethod = step2.validation_method;
      const validationParams = step2.validation_params || {};
      let nSplits = null;
      if (validationMethod === "KFold" || validationMethod === "StratifiedKFold") {
        const candidates = [
          validationParams?.n_splits,
          validationParams?.nSplits,
          validationParams?.folds,
          validationParams?.k,
          validationParams?.nFolds,
          step2?.n_splits,
        ];
        for (const candidate of candidates) {
          const numeric = Number(candidate);
          if (Number.isFinite(numeric) && numeric > 0) {
            nSplits = numeric;
            break;
          }
        }
      }

      const payload = {
        dataset_id: dsId,
        target_name: step2.target,               // backend espera 'target_name'
        mode: step2.classification ? "classification" : "regression",  // backend espera 'mode'
        n_components: step2.n_components,
        threshold: step2.classification && step2.threshold != null ? step2.threshold : undefined,
        n_bootstrap: step2.n_bootstrap ?? 0,
        validation_method: validationMethod,
        validation_params: validationParams,
        n_splits: nSplits ?? undefined,
        spectral_range: spectralRange,
        preprocess: checked,
      };

      const data = await postTrain(payload);
      console.debug('[Step3] rangesStr =', rangesStr);
      console.debug('[Step3] preprocess =', checked);
      console.debug('[Step3] step2 =', step2);
      console.debug('[Step3] data.range_used =', data?.range_used);

      const preprocessSteps = checked.map((method) => ({ method }));
      const fullParams = {
        ...step2,
        ranges: rangesStr,
        spectral_range: spectralRange,
        preprocess_steps: preprocessSteps,
        range_used: data.range_used,
        ...(Number.isFinite(nSplits) ? { n_splits: nSplits } : {}),
      };

      onAnalyzed?.(data, fullParams);
    } catch (err) {
      console.error(err);
      const msg = err?.message || String(err);
      if (msg.includes("415")) {
        alert("Requisição inválida. Envie JSON (Content-Type: application/json).");
      } else {
        alert("Erro ao executar calibração.\n" + msg);
      }
    } finally {
      setRunning(false);
    }
  }

  if (!wavelengths.length) {
    return (
      <div className="bg-white rounded-lg shadow p-6 space-y-3">
        <h2 className="text-lg font-semibold">3. Faixas & Pré-processamento</h2>
        <p className="text-sm text-red-600">
          Metadados indisponíveis. Refaça o upload no passo 1.
        </p>
        <div className="flex gap-2">
          <button type="button" className="bg-gray-200 px-4 py-2 rounded" onClick={onBack}>
            Voltar
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h2 className="text-lg font-semibold">3. Faixas & Pré-processamento</h2>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={showMean}
          onChange={(e) => setShowMean(e.target.checked)}
        />
        Exibir espectro médio
      </label>

      <div ref={chartRef} className="w-full h-80 border rounded bg-white" style={{ minHeight: 320 }} />

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-700 mb-1">λ_min</label>
          <input
            type="number"
            step="any"
            className="w-full border rounded p-2 bg-white text-gray-800"
            value={lambdaMin}
            onChange={(e) => setLambdaMin(e.target.value)}
            min={wlMinAuto || undefined}
            max={wlMaxAuto || undefined}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-700 mb-1">λ_max</label>
          <input
            type="number"
            step="any"
            className="w-full border rounded p-2 bg-white text-gray-800"
            value={lambdaMax}
            onChange={(e) => setLambdaMax(e.target.value)}
            min={wlMinAuto || undefined}
            max={wlMaxAuto || undefined}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="px-4 py-2 rounded bg-gray-100 hover:bg-gray-200" onClick={applyRange}>
          Aplicar faixa
        </button>
        <button type="button" className="px-4 py-2 rounded bg-emerald-100 text-emerald-800 hover:bg-emerald-200" onClick={selectAll}>
          Selecionar tudo
        </button>
        <button type="button" className="px-4 py-2 rounded bg-red-50 text-red-700 hover:bg-red-100" onClick={clearRanges}>
          Limpar
        </button>
        <div className="text-sm ml-2">
          <span className="text-gray-500 mr-1">Selecionada:</span>
          <span className="font-medium">{selectedText}</span>
        </div>
      </div>

      <div>
        <p className="font-semibold mb-2">Pré-processamento</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-y-2">
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="snv" /> SNV</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="msc" /> MSC</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="minmax" /> Normalização Min-Max</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="zscore" /> Z-score</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="sg1" /> 1ª derivada SG</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" name="pp-method" value="sg2" /> 2ª derivada SG</label>
        </div>
      </div>

      <div className="flex gap-2">
        <button type="button" className="bg-gray-200 px-4 py-2 rounded" onClick={onBack}>
          Voltar
        </button>
        <button
          type="button"
          onClick={runAnalysis}
          disabled={running}
          className="bg-emerald-800 hover:bg-emerald-900 text-white px-4 py-2 rounded"
        >
          {running ? "Executando..." : "Executar calibração"}
        </button>
      </div>
    </div>
  );
}
