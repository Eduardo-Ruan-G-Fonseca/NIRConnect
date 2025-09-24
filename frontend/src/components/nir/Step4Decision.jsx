import { useEffect, useMemo, useRef, useState } from "react";
import { postOptimize, postTrain, getOptimizeStatus } from "../../services/api";
import { getDatasetId } from "../../api/http";
import VipTopCard from "./VipTopCard";
import ConfusionMatrixCard from "./ConfusionMatrixCard";
import CvCurveCard from "./CvCurveCard";
import LatentCard from "./LatentCard";
import PerClassMetricsCard from "./PerClassMetricsCard";
import { normalizeTrainResult } from "../../services/normalizeTrainResult";

export default function Step4Decision({ step2, result, dataId, onBack, onContinue }) {
  const baseTrainRes = useMemo(() => result?.data || result || {}, [result]);
  const baseParams = useMemo(
    () => ({ ...(step2 || {}), ...((result && result.params) || {}) }),
    [step2, result]
  );

  const [trainRes, setTrainRes] = useState(baseTrainRes);
  const [currentParams, setCurrentParams] = useState(baseParams);
  const [optLoading, setOptLoading] = useState(false);
  const [optStatus, setOptStatus] = useState({ current: 0, total: 0 });
  const [statusSamples, setStatusSamples] = useState([]);
  const [etaCountdown, setEtaCountdown] = useState(null);
  const [lastDuration, setLastDuration] = useState(null);
  const [lastSummary, setLastSummary] = useState(null);
  const [bestInfo, setBestInfo] = useState(null);
  const [goalInfo, setGoalInfo] = useState(null);
  const [goalWarning, setGoalWarning] = useState(null);
  const optStartRef = useRef(null);
  const lastStatusRef = useRef({ current: 0, total: 0 });
  const etaBaseRef = useRef({ seconds: null, timestamp: null });

  useEffect(() => {
    setTrainRes(baseTrainRes);
  }, [baseTrainRes]);

  useEffect(() => {
    setCurrentParams(baseParams);
  }, [baseParams]);

  const data = useMemo(() => normalizeTrainResult(trainRes), [trainRes]);

  const metricsTrain = data.metrics || {};
  const metricsValid = data.cv_metrics || data.metrics_valid || {};

  const formatMetric = (metric) => {
    if (!metric) return "";
    const label = String(metric).replace(/_/g, " ");
    return label.charAt(0).toUpperCase() + label.slice(1);
  };

  const normalizeSpectralRange = (value) => {
    if (!value) return null;

    const toNumber = (v) => {
      if (typeof v === "number") return Number.isFinite(v) ? v : null;
      if (v === null || v === undefined) return null;
      const s = String(v).trim().replace(",", ".");
      if (s === "") return null;
      const n = Number(s);
      return Number.isFinite(n) ? n : null;
    };

    if (Array.isArray(value)) {
      if (value.length >= 2 && !Array.isArray(value[0])) {
        const min = toNumber(value[0]);
        const max = toNumber(value[1]);
        if (min !== null && max !== null && min !== max) {
          return { min: Math.min(min, max), max: Math.max(min, max) };
        }
      }
      for (const entry of value) {
        if (Array.isArray(entry) && entry.length >= 2) {
          const min = toNumber(entry[0]);
          const max = toNumber(entry[1]);
          if (min !== null && max !== null && min !== max) {
            return { min: Math.min(min, max), max: Math.max(min, max) };
          }
        }
      }
      return null;
    }

    if (typeof value === "object") {
      const min = toNumber(value.min ?? value[0]);
      const max = toNumber(value.max ?? value[1]);
      if (min !== null && max !== null && min !== max) {
        return { min: Math.min(min, max), max: Math.max(min, max) };
      }
      return null;
    }

    if (typeof value === "string") {
      const candidates = value.split(",");
      for (const candidate of candidates) {
        const [a, b] = candidate.split(/[-–—]/);
        const min = toNumber(a);
        const max = toNumber(b);
        if (min !== null && max !== null && min !== max) {
          return { min: Math.min(min, max), max: Math.max(min, max) };
        }
      }
    }

    return null;
  };

  const normalizePreprocessGrid = (grid, fallbackPipeline) => {
    const safeFallback = Array.isArray(fallbackPipeline)
      ? fallbackPipeline.map((step) => (typeof step === "string" ? step : step?.method)).filter(Boolean)
      : [];

    if (Array.isArray(grid) && grid.length) {
      const mapped = grid
        .map((pipeline) => {
          if (Array.isArray(pipeline)) {
            const steps = pipeline
              .map((step) => (typeof step === "string" ? step : step?.method))
              .filter((step) => typeof step === "string" && step);
            return steps.length ? steps : [];
          }
          if (typeof pipeline === "string" && pipeline) {
            return [pipeline];
          }
          return [];
        })
        .filter((pipeline) => Array.isArray(pipeline));
      if (mapped.length) {
        return mapped;
      }
    }

    if (safeFallback.length) {
      return [safeFallback];
    }

    return [[]];
  };

  const normalizeSgParams = (params) => {
    if (!Array.isArray(params)) return [];
    return params
      .map((entry) => {
        if (Array.isArray(entry)) {
          const [w, p, d] = entry;
          const parsed = [w, p, d].map((value) => (Number.isFinite(value) ? value : Number(value)));
          return parsed.every((value) => Number.isFinite(value)) ? parsed.map((value) => Math.trunc(value)) : null;
        }
        if (entry && typeof entry === "object") {
          const window = entry.window ?? entry.window_length ?? entry[0];
          const poly = entry.poly ?? entry.polyorder ?? entry[1];
          const deriv = entry.deriv ?? entry.derivative ?? entry[2];
          const parsed = [window, poly, deriv].map((value) => (Number.isFinite(value) ? value : Number(value)));
          return parsed.every((value) => Number.isFinite(value)) ? parsed.map((value) => Math.trunc(value)) : null;
        }
        return null;
      })
      .filter(Boolean);
  };

  function MetricsCard({ title, metrics }) {
    const entries = Object.entries(metrics || {});
    if (!entries.length) return (
      <div className="card dashed h-64 flex items-center justify-center"><p>Sem dados.</p></div>
    );
    return (
      <div className="card p-4" style={{ minHeight: 420 }}>
        <h3 className="card-title mb-3">{title}</h3>
        <div className="overflow-x-auto" style={{ maxHeight: 360, overflowY: "auto" }}>
          <table className="table table-sm">
            <tbody>
              {entries.map(([k, v]) => (
                <tr key={k}>
                  <th>{k}</th>
                  <td>{Number.isFinite(v) ? Number(v).toFixed(4) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  async function handleOptimize() {
    const ds = dataId || getDatasetId();
    if (!ds) {
      alert("Dataset não encontrado — volte ao passo 1 e faça o upload.");
      return;
    }

    const combinedParams = { ...(step2 || {}), ...(currentParams || {}) };
    const validationMethod = combinedParams.validation_method;
    const validationParams = combinedParams.validation_params || {};

    const extractNSplits = () => {
      if (validationMethod !== "KFold" && validationMethod !== "StratifiedKFold") {
        const fallback = Number(combinedParams?.n_splits);
        return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
      }
      const candidates = [
        validationParams?.n_splits,
        validationParams?.nSplits,
        validationParams?.folds,
        validationParams?.k,
        validationParams?.nFolds,
        combinedParams?.n_splits,
      ];
      for (const value of candidates) {
        const num = Number(value);
        if (Number.isFinite(num) && num > 0) {
          return num;
        }
      }
      return null;
    };

    const nSplits = extractNSplits();
    const threshold =
      combinedParams.classification && combinedParams.threshold != null
        ? combinedParams.threshold
        : undefined;
    const kMaxCandidate =
      combinedParams?.k_max ??
      combinedParams?.n_components_max ??
      combinedParams?.n_components ??
      null;
    const parsedKMax =
      kMaxCandidate !== null && kMaxCandidate !== undefined ? Number(kMaxCandidate) : null;
    const kMax =
      parsedKMax !== null && Number.isFinite(parsedKMax) && parsedKMax > 0 ? parsedKMax : null;

    const preprocessList = Array.isArray(combinedParams?.preprocess_steps)
      ? combinedParams.preprocess_steps.map((p) => (typeof p === "string" ? p : p?.method))
      : Array.isArray(combinedParams?.preprocess)
      ? combinedParams.preprocess
      : [];
    const spectralRange =
      normalizeSpectralRange(combinedParams?.spectral_range) ||
      normalizeSpectralRange(combinedParams?.spectral_ranges) ||
      normalizeSpectralRange(combinedParams?.ranges) ||
      normalizeSpectralRange(combinedParams?.range_used);
    const preprocessGrid = normalizePreprocessGrid(
      combinedParams?.preprocess_grid,
      preprocessList.filter(Boolean)
    );
    const nBootstrap = Number(combinedParams?.n_bootstrap ?? 0) || 0;
    const minScoreValue = Number(combinedParams?.min_score);
    const hasMinScore = Number.isFinite(minScoreValue);

    optStartRef.current = Date.now();
    setLastDuration(null);
    setLastSummary(null);
    setStatusSamples([]);
    setEtaCountdown(null);
    etaBaseRef.current = { seconds: null, timestamp: null };
    setOptStatus({ current: 0, total: 0 });
    setOptLoading(true);
    setGoalInfo(null);
    setGoalWarning(null);
    try {
      const optPayload = {
        dataset_id: ds,
        target_name: combinedParams.target,
        mode: combinedParams.classification ? "classification" : "regression",
        validation_method: validationMethod,
        threshold,
        k_min: 1,
        k_max: kMax,
      };
      if (spectralRange) {
        optPayload.spectral_range = spectralRange;
      }
      if (Number.isFinite(nSplits)) {
        optPayload.n_splits = nSplits;
      }
      if (combinedParams.metric_goal) {
        optPayload.metric_goal = combinedParams.metric_goal;
      }
      if (hasMinScore) {
        optPayload.min_score = minScoreValue;
      }
      optPayload.preprocess_grid = preprocessGrid;

      const sgCandidates =
        Array.isArray(combinedParams?.sg_params) && combinedParams.sg_params.length
          ? combinedParams.sg_params
          : Array.isArray(combinedParams?.sg_grid)
          ? combinedParams.sg_grid
          : [];
      const normalizedSg = normalizeSgParams(sgCandidates);
      const hasSgMethod = preprocessGrid.some((pipeline) => pipeline.some((method) => method?.startsWith("sg")));
      if (!hasSgMethod) {
        optPayload.sg_params = [];
      } else if (normalizedSg.length) {
        optPayload.sg_params = normalizedSg;
      }

      const opt = await postOptimize(optPayload);

      const bestParams = opt?.best?.params || opt?.best_params || {};
      const bestReport =
        opt?.best?.report && typeof opt.best.report === "object" ? opt.best.report : {};
      const bestK = bestParams?.n_components;
      const bestPreprocess = Array.isArray(bestParams?.preprocess)
        ? bestParams.preprocess.filter(Boolean)
        : [];
      const [bestSgCandidate] = normalizeSgParams(
        bestParams?.sg != null ? [bestParams.sg] : []
      );
      const fallbackPreprocess = preprocessList.filter(Boolean);
      const finalPreprocess = bestPreprocess.length ? bestPreprocess : fallbackPreprocess;
      const finalPreprocessGrid = finalPreprocess.length
        ? [finalPreprocess]
        : preprocessGrid;
      const defaultSgCandidate = Array.isArray(normalizedSg) && normalizedSg.length ? normalizedSg[0] : null;
      const finalSgTuple = bestSgCandidate || defaultSgCandidate || null;
      const usesSgMethod = finalPreprocess.some((method) => method?.startsWith("sg"));
      const sgForTraining = usesSgMethod && finalSgTuple ? finalSgTuple : null;
      if (bestK) {
        const trained = await postTrain({
          dataset_id: ds,
          target_name: combinedParams.target,
          mode: combinedParams.classification ? "classification" : "regression",
          validation_method: validationMethod,
          validation_params: validationParams,
          n_splits: nSplits ?? undefined,
          n_bootstrap: nBootstrap,
          threshold,
          n_components: bestK,
          preprocess: finalPreprocess,
          spectral_range: spectralRange,
          preprocess_grid: finalPreprocessGrid,
          sg: sgForTraining,
        });
        const enriched = {
          ...trained,
          ...bestReport,
        };
        if (Object.keys(bestReport || {}).length) {
          enriched.best_report = bestReport;
        }
        if (!enriched.residuals && trained?.residuals) {
          enriched.residuals = trained.residuals;
        }
        if (!enriched.influence && trained?.influence) {
          enriched.influence = trained.influence;
        }
        if (!enriched.distributions && trained?.distributions) {
          enriched.distributions = trained.distributions;
        }
        if (!enriched.predictions && trained?.predictions) {
          enriched.predictions = trained.predictions;
        }
        setTrainRes(enriched);
        const scoreValue = opt?.best?.score ?? opt?.best_score ?? null;
        setBestInfo({
          k: bestK,
          score: scoreValue,
          preprocess: finalPreprocess,
          sg: sgForTraining,
          usesSg: usesSgMethod,
        });
        const fallbackCombos = (preprocessGrid.length || 1) * (kMax || bestK || 1);
        const totalCombos =
          opt?.total_combinations ??
          opt?.total ??
          opt?.trials ??
          optStatus.total ??
          fallbackCombos;
        setLastSummary({
          totalCombos: Number.isFinite(totalCombos) && totalCombos > 0 ? totalCombos : null,
          finishedAt: new Date(),
        });
        setGoalInfo(opt?.goal || null);
        setGoalWarning(opt?.goal_warning || null);
        setCurrentParams(() => {
          const fallbackSteps = Array.isArray(combinedParams?.preprocess_steps)
            ? combinedParams.preprocess_steps
            : fallbackPreprocess.map((method) => ({ method }));
          const fallbackGrid = preprocessGrid.length
            ? preprocessGrid
            : normalizePreprocessGrid(combinedParams?.preprocess_grid, fallbackPreprocess);
          const nextPreprocessSteps = finalPreprocess.length
            ? finalPreprocess.map((method) => ({ method }))
            : fallbackSteps;
          const nextPreprocessGrid = finalPreprocess.length ? [finalPreprocess] : fallbackGrid;
          const nextSgParams = usesSgMethod && finalSgTuple ? [finalSgTuple] : [];
          return {
            ...combinedParams,
            n_components: bestK,
            optimized: true,
            best_score: scoreValue,
            best_params: {
              ...bestParams,
              preprocess: finalPreprocess,
              sg: sgForTraining,
            },
            best_report: bestReport,
            min_score: hasMinScore ? minScoreValue : combinedParams?.min_score,
            range_used: trained?.range_used ?? combinedParams?.range_used,
            preprocess: finalPreprocess,
            preprocess_steps: nextPreprocessSteps,
            preprocess_grid: nextPreprocessGrid,
            sg: sgForTraining || undefined,
            sg_params: nextSgParams,
            spectral_range: spectralRange ?? combinedParams?.spectral_range,
            thresholds: bestReport?.thresholds ?? combinedParams?.thresholds,
            n_selected_variables:
              bestReport?.n_selected_variables ?? combinedParams?.n_selected_variables,
          };
        });
        document.getElementById('cv-curve')?.scrollIntoView({ behavior: 'smooth' });
      } else {
        alert('Não foi possível sugerir k.');
      }
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e?.data?.detail ?? e?.message ?? e;
      const msg =
        typeof raw === "string"
          ? raw
          : raw?.message
          ? `${raw.message}\n${JSON.stringify(raw, null, 2)}`
          : JSON.stringify(raw, null, 2);
      alert(msg);
    } finally {
      setOptLoading(false);
      if (optStartRef.current) {
        setLastDuration(Date.now() - optStartRef.current);
        optStartRef.current = null;
      }
    }
  }

  useEffect(() => {
    let timeoutId;
    let cancelled = false;
    let controller;
    let delay = 1000;

    const run = async () => {
      if (cancelled || !optLoading) {
        return;
      }

      if (controller) {
        controller.abort();
      }
      controller = new AbortController();

      let nextDelay = Math.min(delay * 1.5, 5000);
      let aborted = false;

      try {
        const status = await getOptimizeStatus({ signal: controller.signal });
        if (cancelled || !optLoading) {
          return;
        }

        const current = Number(status.current) || 0;
        const total = Number(status.total) || 0;
        const prevStatus = lastStatusRef.current;
        const changed = prevStatus.current !== current || prevStatus.total !== total;

        if (changed) {
          lastStatusRef.current = { current, total };
        }

        setOptStatus((prev) => {
          if (prev.current === current && prev.total === total) {
            return prev;
          }
          return { current, total };
        });

        setStatusSamples((prev) => {
          const sample = { time: Date.now(), current };
          const next = [...prev, sample];
          return next.slice(-10);
        });

        if (changed) {
          nextDelay = 1000;
        }
      } catch (err) {
        if (err?.name === "AbortError") {
          aborted = true;
        } else if (!cancelled) {
          console.warn("Não foi possível consultar o status da otimização", err);
          nextDelay = Math.min(delay * 1.5, 10000);
        }
      } finally {
        const shouldSchedule = !cancelled && optLoading && !aborted;
        if (shouldSchedule) {
          delay = nextDelay;
          timeoutId = setTimeout(run, delay);
        }
      }
    };

    if (optLoading) {
      delay = 1000;
      run();
    } else {
      if (controller) {
        controller.abort();
      }
      lastStatusRef.current = { current: 0, total: 0 };
      setOptStatus({ current: 0, total: 0 });
      setStatusSamples([]);
      setEtaCountdown(null);
      etaBaseRef.current = { seconds: null, timestamp: null };
    }

    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (controller) {
        controller.abort();
      }
    };
  }, [optLoading]);

  useEffect(() => {
    if (!optLoading) {
      setEtaCountdown(null);
      etaBaseRef.current = { seconds: null, timestamp: null };
      return;
    }

    const { current, total } = optStatus;
    if (!total) {
      setEtaCountdown(null);
      etaBaseRef.current = { seconds: null, timestamp: null };
      return;
    }

    if (current >= total) {
      setEtaCountdown(0);
      etaBaseRef.current = { seconds: 0, timestamp: Date.now() };
      return;
    }

    if (statusSamples.length < 2) {
      return;
    }

    const first = statusSamples[0];
    const last = statusSamples[statusSamples.length - 1];
    const deltaC = last.current - first.current;
    const deltaT = (last.time - first.time) / 1000;
    if (deltaC <= 0 || deltaT <= 0) {
      return;
    }

    const remaining = Math.max(0, total - last.current);
    if (!remaining) {
      setEtaCountdown(0);
      etaBaseRef.current = { seconds: 0, timestamp: Date.now() };
      return;
    }

    const secondsPerCombo = deltaT / deltaC;
    if (!Number.isFinite(secondsPerCombo) || secondsPerCombo <= 0) {
      return;
    }

    const seconds = remaining * secondsPerCombo;
    etaBaseRef.current = { seconds, timestamp: Date.now() };
    setEtaCountdown(seconds);
  }, [optLoading, optStatus, statusSamples]);

  useEffect(() => {
    if (!optLoading) {
      return () => {};
    }

    let cancelled = false;
    let timeoutId;

    const updateCountdown = () => {
      if (cancelled) return;
      const { seconds, timestamp } = etaBaseRef.current;
      if (seconds == null || timestamp == null) {
        timeoutId = setTimeout(updateCountdown, 1000);
        return;
      }
      const elapsed = (Date.now() - timestamp) / 1000;
      const next = Math.max(0, seconds - elapsed);
      setEtaCountdown((prev) => {
        const roundedPrev = prev == null ? null : Math.round(prev * 2) / 2;
        const roundedNext = Math.round(next * 2) / 2;
        if (roundedPrev === roundedNext) {
          return prev;
        }
        return next;
      });
      timeoutId = setTimeout(updateCountdown, 1000);
    };

    timeoutId = setTimeout(updateCountdown, 1000);

    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [optLoading, statusSamples]);

  const progressPercent = useMemo(() => {
    if (!optLoading) return null;
    const { current, total } = optStatus;
    if (total > 0) {
      return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
    }
    return 5;
  }, [optLoading, optStatus]);

  const formatDuration = (value, isMilliseconds = false) => {
    if (value == null) return null;
    const seconds = isMilliseconds ? Math.round(value / 1000) : Math.ceil(value);
    const safeSeconds = Math.max(0, seconds);
    const minutes = Math.floor(safeSeconds / 60);
    const rem = safeSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  };

  function handleBack() {
    onBack?.(trainRes, currentParams);
  }

  function handleContinue() {
    onContinue?.(trainRes, currentParams);
  }

  return (
    <div className="space-y-6">
      {goalWarning && (
        <div className="border border-amber-300 bg-amber-50 text-amber-900 rounded-lg p-4">
          <h3 className="font-semibold text-sm">Meta de desempenho não atingida</h3>
          <p className="text-sm mt-1">{goalWarning}</p>
          <ul className="list-disc pl-5 text-xs mt-3 space-y-1">
            <li>Experimente aumentar o número de componentes latentes avaliados.</li>
            <li>Teste combinações adicionais de pré-processamentos espectrais.</li>
            <li>Considere coletar mais dados representativos ou revisar a rotulagem.</li>
          </ul>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsCard title="Treino" metrics={metricsTrain} />
        <MetricsCard title="Validação" metrics={metricsValid} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VipTopCard vip={data.vip} top={30} />
        <ConfusionMatrixCard cm={data.cm} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CvCurveCard curve={data.cv_curve} task={data.task} />
        <LatentCard latent={data.latent} labels={data.latent?.sample_labels} />
      </div>

      <PerClassMetricsCard perClass={data.per_class} />

      {optLoading && (
        <div className="card p-4 bg-slate-50 border border-slate-200">
          <div className="flex items-center justify-between text-sm text-slate-700 mb-2">
            <span>
              Otimizando combinações
              {optStatus.total ? ` (${optStatus.current} de ${optStatus.total})` : "..."}
            </span>
            {etaCountdown !== null && etaCountdown !== undefined && (
              <span>
                Tempo estimado: {etaCountdown === 0 ? "quase lá" : `~${formatDuration(etaCountdown)}`}
              </span>
            )}
          </div>
          <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-500"
              style={{ width: `${progressPercent ?? 5}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          onClick={handleOptimize}
          className="bg-blue-600 hover:bg-blue-700 text-white py-3 px-5 rounded-lg shadow-sm flex items-center justify-center disabled:opacity-50"
          disabled={optLoading}
        >
          {optLoading ? <i className="fas fa-spinner fa-spin mr-2"></i> : <i className="fas fa-cogs mr-2"></i>}
          Otimizar Modelo
        </button>

        {bestInfo && (
          <div className="text-sm text-gray-700">
            Modelo otimizado: k = {bestInfo.k}
            {bestInfo.score != null && Number.isFinite(Number(bestInfo.score)) && (
              <> (score: {Number(bestInfo.score).toFixed(3)})</>
            )}
            {goalInfo?.target != null && (
              <>
                {" "}— meta ({formatMetric(goalInfo.metric || "balanced_accuracy")} {goalInfo.comparison || ">="}{" "}
                {goalInfo.target.toFixed(3)})
              </>
            )}
          </div>
        )}

        {!optLoading && lastDuration != null && (
          <div className="text-xs text-slate-600">
            Última otimização: {formatDuration(lastDuration, true)}
            {lastSummary?.totalCombos ? ` • ${lastSummary.totalCombos} combinações avaliadas` : ""}
            {lastSummary?.finishedAt
              ? ` • concluída às ${lastSummary.finishedAt.toLocaleTimeString()}`
              : ""}
          </div>
        )}

        {Array.isArray(bestInfo?.preprocess) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="card p-4">
              <h3 className="card-title mb-2">Pré-processamento vencedor</h3>
              <p className="text-sm text-gray-800">
                {bestInfo.preprocess.length
                  ? bestInfo.preprocess.join(" + ")
                  : "Sem pré-processamento espectral"}
              </p>
              {bestInfo.sg && (
                <p className="text-xs text-gray-600 mt-2">
                  SG: janela {bestInfo.sg[0]}, ordem {bestInfo.sg[1]}, derivada {bestInfo.sg[2]}
                </p>
              )}
              {bestInfo.usesSg && !bestInfo.sg && (
                <p className="text-xs text-gray-600 mt-2">
                  SG: parâmetros padrão (janela 11, ordem 2, derivada 1)
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-between">
        <button
          onClick={handleBack}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-lg"
        >
          Voltar
        </button>
        <button
          onClick={handleContinue}
          className="bg-emerald-700 hover:bg-emerald-800 text-white py-2 px-4 rounded-lg"
        >
          Continuar
        </button>
      </div>
    </div>
  );
}

