import { useEffect, useMemo, useState } from "react";
import { postOptimize, postTrain } from "../../services/api";
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
  const [bestInfo, setBestInfo] = useState(null);
  const [goalInfo, setGoalInfo] = useState(null);
  const [goalWarning, setGoalWarning] = useState(null);

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
    const spectralRanges =
      combinedParams?.spectral_ranges ??
      combinedParams?.ranges ??
      combinedParams?.spectral_range ??
      null;
    const nBootstrap = Number(combinedParams?.n_bootstrap ?? 0) || 0;
    const minScoreValue = Number(combinedParams?.min_score);
    const hasMinScore = Number.isFinite(minScoreValue);

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
      if (Number.isFinite(nSplits)) {
        optPayload.n_splits = nSplits;
      }
      if (combinedParams.metric_goal) {
        optPayload.metric_goal = combinedParams.metric_goal;
      }
      if (hasMinScore) {
        optPayload.min_score = minScoreValue;
      }

      const opt = await postOptimize(optPayload);

      const bestParams = opt?.best?.params || opt?.best_params || {};
      const bestK = bestParams?.n_components;
      if (bestK) {
        const trained = await postTrain({
          dataset_id: ds,
          target_name: combinedParams.target,
          mode: combinedParams.classification ? "classification" : "regression",
          validation_method: validationMethod,
          validation_params: validationParams,
          n_bootstrap: nBootstrap,
          threshold,
          n_components: bestK,
          preprocess: preprocessList.filter(Boolean),
          spectral_ranges: spectralRanges,
        });
        setTrainRes(trained);
        const scoreValue = opt?.best?.score ?? opt?.best_score ?? null;
        setBestInfo({ k: bestK, score: scoreValue });
        setGoalInfo(opt?.goal || null);
        setGoalWarning(opt?.goal_warning || null);
        setCurrentParams(() => {
          const nextPreprocessSteps = Array.isArray(combinedParams?.preprocess_steps)
            ? combinedParams.preprocess_steps
            : preprocessList.filter(Boolean).map((method) => ({ method }));
          return {
            ...combinedParams,
            n_components: bestK,
            optimized: true,
            best_score: scoreValue,
            best_params: bestParams,
            min_score: hasMinScore ? minScoreValue : combinedParams?.min_score,
            range_used: trained?.range_used ?? combinedParams?.range_used,
            preprocess_steps: nextPreprocessSteps,
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
    }
  }

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

