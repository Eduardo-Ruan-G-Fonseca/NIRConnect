import { useMemo, useState } from "react";
import { postOptimize, postTrain } from "../../services/api";
import { getDatasetId } from "../../api/http";
import VipTopCard from "./VipTopCard";
import ConfusionMatrixCard from "./ConfusionMatrixCard";
import CvCurveCard from "./CvCurveCard";
import LatentCard from "./LatentCard";
import PerClassMetricsCard from "./PerClassMetricsCard";
import { normalizeTrainResult } from "../../services/normalizeTrainResult";

export default function Step4Decision({ step2, result, dataId }) {
  const [trainRes, setTrainRes] = useState(result?.data || result || {});
  const params = result?.params || {};
  const [optLoading, setOptLoading] = useState(false);
  const [bestInfo, setBestInfo] = useState(null);

  const data = useMemo(() => normalizeTrainResult(trainRes), [trainRes]);

  const metricsTrain = data.metrics || {};
  const metricsValid = data.cv_metrics || data.metrics_valid || {};

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
    setOptLoading(true);
    try {
      const opt = await postOptimize({
        dataset_id: ds,
        target_name: step2.target,
        mode: step2.classification ? "classification" : "regression",
        validation_method: step2.validation_method,
        n_splits: step2.n_splits,
        threshold: step2.threshold,
        k_min: 1,
        k_max: step2.n_components_max || null,
      });

      const bestK = opt?.best_params?.n_components;
      if (bestK) {
        const trained = await postTrain({
          dataset_id: ds,
          target_name: step2.target,
          mode: step2.classification ? "classification" : "regression",
          validation_method: step2.validation_method,
          n_splits: step2.n_splits,
          threshold: step2.threshold,
          n_components: bestK,
          preprocess: params.preprocess_steps?.map(p => p.method) || [],
          spectral_ranges: params.ranges || null,
        });
        setTrainRes(trained);
        setBestInfo({ k: bestK, score: opt?.best_score });
        document.getElementById('cv-curve')?.scrollIntoView({ behavior: 'smooth' });
      } else {
        alert('Não foi possível sugerir k.');
      }
    } catch (e) {
      console.error(e);
      alert('Falha ao otimizar.');
    } finally {
      setOptLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsCard title="Treino" metrics={metricsTrain} />
        <MetricsCard title="Validação" metrics={metricsValid} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VipTopCard vip={data.vip} top={30} />
        <ConfusionMatrixCard cm={data.cm} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CvCurveCard curve={data.cv_curve} task={data.task} recommended={data.recommended_n_components} />
        <LatentCard latent={data.latent} labels={data.latent?.sample_labels} />
      </div>

      <PerClassMetricsCard perClass={data.per_class} />

      <button
        onClick={handleOptimize}
        className="bg-blue-600 hover:bg-blue-700 text-white py-3 px-5 rounded-lg shadow-sm flex items-center disabled:opacity-50"
        disabled={optLoading}
      >
        {optLoading ? <i className="fas fa-spinner fa-spin mr-2"></i> : <i className="fas fa-cogs mr-2"></i>}
        Otimizar Modelo
      </button>

      {bestInfo && (
        <div className="mt-4 text-sm text-gray-700">
          Modelo otimizado: k = {bestInfo.k} (score: {bestInfo.score?.toFixed?.(3) ?? bestInfo.score})
        </div>
      )}
    </div>
  );
}

