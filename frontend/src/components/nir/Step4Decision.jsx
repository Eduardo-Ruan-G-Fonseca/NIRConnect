import { useMemo, useState } from "react";
import { postOptimize } from "../../services/api";
import { getDatasetId } from "../../api/http";
import VipTopCard from "./VipTopCard";
import ConfusionMatrixCard from "./ConfusionMatrixCard";
import CvCurveCard from "./CvCurveCard";
import LatentCard from "./LatentCard";
import PerClassMetricsCard from "./PerClassMetricsCard";
import { normalizeTrainResult } from "../../services/normalizeTrainResult";

export default function Step4Decision({ step2, result }) {
  const [optimizeResult, setOptimizeResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const data = useMemo(() => normalizeTrainResult(result?.data || result || {}), [result]);

  async function handleOptimize() {
    if (!getDatasetId()) {
      alert("Dataset não encontrado — volte ao passo 1 e faça o upload.");
      return;
    }
    setBusy(true);
    try {
      const res = await postOptimize({
        mode: step2?.classification ? "classification" : "regression",
        target_name: step2.target,
        threshold: step2.threshold,
        validation_method: step2.validation_method,
        n_splits: step2.n_splits,
        k_min: 1,
        k_max: step2.n_components_max,
      });
      setOptimizeResult(res);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VipTopCard vip={data.vip} top={30} />
        <ConfusionMatrixCard cm={data.cm} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CvCurveCard
          curve={optimizeResult?.curve || data.cv_curve}
          task={data.task}
          recommended={optimizeResult?.best_params?.n_components}
        />
        <LatentCard latent={data.latent} labels={data.latent?.sample_labels} />
      </div>

      <PerClassMetricsCard perClass={data.per_class} />

      {!optimizeResult && (
        <button
          onClick={handleOptimize}
          className="bg-blue-600 hover:bg-blue-700 text-white py-3 px-5 rounded-lg shadow-sm flex items-center disabled:opacity-50"
          disabled={busy}
        >
          {busy ? <i className="fas fa-spinner fa-spin mr-2"></i> : <i className="fas fa-cogs mr-2"></i>}
          Otimizar Modelo
        </button>
      )}

      {optimizeResult && (
        <div className="mt-4 text-sm text-gray-700">
          Melhor n_components: {optimizeResult.best_params?.n_components} (score: {optimizeResult.best_score?.toFixed?.(3) ?? optimizeResult.best_score})
        </div>
      )}
    </div>
  );
}
