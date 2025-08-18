// src/components/nir/Step2Parameters.jsx
import { useEffect, useState } from "react";
import { getJSON } from "../../api/http";

/** Controle inteiro com ±, input e slider */
function NumberChooser({
  label,
  min = 0,
  max = 100,
  step = 1,
  value,
  onChange,
  hint = "Digite, use ± ou arraste a barra.",
}) {
  const clamp = (v) => {
    const n = Number(v);
    if (Number.isNaN(n)) return value ?? min;
    return Math.min(max, Math.max(min, n));
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>

      {/* Indicador verde acima */}
      <div className="flex items-center gap-2 text-sm text-emerald-700 mb-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-600" />
        <span>{hint}</span>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          className="px-3 py-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
          onClick={() => onChange(clamp(value - step))}
        >
          –
        </button>

        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(clamp(e.target.value))}
          className="w-28 border rounded p-2 bg-white !text-gray-800 !border-gray-300"
        />

        <button
          type="button"
          className="px-3 py-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
          onClick={() => onChange(clamp(value + step))}
        >
          +
        </button>
      </div>

      {/* Slider */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(clamp(e.target.value))}
        className="w-full mt-2 accent-emerald-600"
      />
    </div>
  );
}

/** Controle decimal com ±, input e slider */
function DecimalChooser({
  label,
  min = 0,
  max = 1,
  step = 0.01,
  value,
  onChange,
  hint = "Digite, use ± ou arraste a barra.",
}) {
  const clamp = (v) => {
    const n = Number(v);
    if (Number.isNaN(n)) return value ?? min;
    return Math.min(max, Math.max(min, n));
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>

      {/* Indicador verde acima */}
      <div className="flex items-center gap-2 text-sm text-emerald-700 mb-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-600" />
        <span>{hint}</span>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          className="px-3 py-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
          onClick={() => onChange(clamp((value - step).toFixed(3)))}
        >
          –
        </button>

        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(clamp(e.target.value))}
          className="w-28 border rounded p-2 bg-white !text-gray-800 !border-gray-300"
        />

        <button
          type="button"
          className="px-3 py-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
          onClick={() => onChange(clamp((value + step).toFixed(3)))}
        >
          +
        </button>
      </div>

      {/* Slider */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(clamp(e.target.value))}
        className="w-full mt-2 accent-emerald-600"
      />
    </div>
  );
}

export default function Step2Parameters({ onBack, onNext }) {
  const [targets, setTargets] = useState([]);
  const [features, setFeatures] = useState([]);
  const [error, setError] = useState("");

  const [target, setTarget] = useState("");
  const [nComponents, setNComponents] = useState(5);
  const [classification, setClassification] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [nBootstrap, setNBootstrap] = useState(0);
  const [validationMethod, setValidationMethod] = useState("KFold");
  const [kfoldSplits, setKfoldSplits] = useState(5);
  const [testSize, setTestSize] = useState(0.3);

  useEffect(() => {
    (async () => {
      try {
        let ts = JSON.parse(localStorage.getItem("nir.targets") || "[]");
        let cols = JSON.parse(localStorage.getItem("nir.columns") || "[]");
        const dsid = localStorage.getItem("nir.datasetId");
        if ((!ts || ts.length === 0) && dsid) {
          const meta = await getJSON(`/columns/meta?dataset_id=${encodeURIComponent(dsid)}`);
          ts = meta.targets || [];
          cols = meta.columns || cols;
        }
        setTargets(ts || []);
        setFeatures(cols || []);
        if (!target && (ts || []).length) {
          setTarget(ts[0]);
        }
      } catch (e) {
        setError(e.message || "Falha ao buscar colunas");
        setTargets([]);
        setFeatures([]);
      }
    })();
  }, []);

  function handleSubmit(e) {
    e.preventDefault();
    if (!target) {
      alert("Selecione a variável-alvo");
      return;
    }
    const params = {
      target,
      n_components: Number(nComponents),
      classification: Boolean(classification),
      threshold: classification ? Number(threshold) : undefined,
      n_bootstrap: Number(nBootstrap),
      validation_method: validationMethod,
      validation_params:
        validationMethod === "KFold"
          ? { n_splits: Number(kfoldSplits), shuffle: true, random_state: 42 }
          : validationMethod === "Holdout"
          ? { test_size: Number(testSize) }
          : {},
    };
    onNext(params);
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <h2 className="text-lg font-semibold">2. Parâmetros</h2>

      {error && <div className="text-sm text-red-600">{error}</div>}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* alvo + modo de análise */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Variável-alvo
            </label>
            <select
              className="mt-1 w-full border rounded p-2 bg-white !text-gray-800 !border-gray-300"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            >
              {targets.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Modo de Análise
            </label>
            <select
              className="mt-1 w-full border rounded p-2 bg-white !text-gray-800 !border-gray-300"
              value={classification ? "true" : "false"}
              onChange={(e) => setClassification(e.target.value === "true")}
            >
              <option value="false">Regressão (PLS-R)</option>
              <option value="true">Classificação (PLS-DA)</option>
            </select>
          </div>
        </div>

        {/* nº de componentes + bootstrap */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <NumberChooser
            label="Nº de Componentes"
            min={1}
            max={50}
            step={1}
            value={nComponents}
            onChange={setNComponents}
          />

          <NumberChooser
            label="Nº de bootstrap"
            min={0}
            max={500}
            step={10}
            value={nBootstrap}
            onChange={setNBootstrap}
            hint="0 desativa. Recomendado 50–200 para estabilidade."
          />
        </div>

        {/* threshold (se classificação) */}
        {classification && (
          <DecimalChooser
            label="Threshold de classificação (0.00–1.00)"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={setThreshold}
          />
        )}

        {/* validação */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Método de Validação
          </label>
          <select
            className="mt-1 w-full border rounded p-2 bg-white !text-gray-800 !border-gray-300"
            value={validationMethod}
            onChange={(e) => setValidationMethod(e.target.value)}
          >
            <option value="KFold">K-Fold</option>
            {classification && <option value="StratifiedKFold">Stratified K-Fold</option>}
            <option value="LOO">Leave-One-Out (LOO)</option>
            <option value="Holdout">Train/Test Split</option>
          </select>
          {validationMethod === "LOO" && (
            <p className="mt-1 text-xs text-amber-700">
              LOO executa 1 treino por amostra (pode ser lento em bases grandes).
            </p>
          )}
        </div>

        {/* params de validação condicionais */}
        {validationMethod === "KFold" && (
          <NumberChooser
            label="Número de folds (k)"
            min={2}
            max={20}
            step={1}
            value={kfoldSplits}
            onChange={setKfoldSplits}
            hint="Escolha k (recomendado 5–10)."
          />
        )}

        {validationMethod === "Holdout" && (
          <DecimalChooser
            label="Proporção de teste"
            min={0.1}
            max={0.9}
            step={0.05}
            value={testSize}
            onChange={setTestSize}
            hint="Ex.: 0.3 → 70% treino / 30% teste."
          />
        )}

        <div className="flex gap-2">
          <button
            type="button"
            className="bg-gray-200 px-4 py-2 rounded"
            onClick={onBack}
          >
            Voltar
          </button>
          <button
            type="submit"
            className="bg-emerald-800 hover:bg-emerald-900 text-white px-4 py-2 rounded"
          >
            Próximo
          </button>
        </div>
      </form>
    </div>
  );
}
