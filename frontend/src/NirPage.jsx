import { useState } from "react";
import Step1Upload from "./components/nir/Step1Upload.jsx";
import Step2Parameters from "./components/nir/Step2Parameters.jsx";
import Step3Preprocess from "./components/nir/Step3Preprocess.jsx";
import Step4Decision from "./components/nir/Step4Decision.jsx";
import Step5Result from "./components/nir/Step5Result.jsx";

export default function NirPage() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [meta, setMeta] = useState(null);
  const [step2, setStep2] = useState(null);
  const [result, setResult] = useState(null); // { data, params }

  function resetAll() {
    setStep(1);
    setFile(null);
    setMeta(null);
    setStep2(null);
    setResult(null);
  }

  return (
    <div className="w-full max-w-[1400px] mx-auto px-4 md:px-6 lg:px-8 space-y-6">
      {/* Cabeçalho */}
      <header className="text-center space-y-1">
        <h1 className="text-3xl font-semibold text-[#2e5339]">
          Plataforma NIR para Aplicações Florestais
        </h1>
        <p className="text-lg text-[#305e6b]">
          Modelagem de Propriedades da Madeira via Espectroscopia NIR
        </p>
      </header>

      {/* Progresso */}
      <nav className="text-sm text-gray-500 flex flex-wrap gap-2 justify-center md:justify-between">
        {["1. Upload","2. Parâmetros","3. Pré-processamento","4. Decisão","5. Resultado"]
          .map((t, i) => (
            <span key={t} className={`step ${i < step ? "font-semibold text-green-700" : ""}`}>
              {t}
            </span>
        ))}
      </nav>

      {/* STEP 1 */}
      {step === 1 && (
        <Step1Upload
          onSuccess={({ file, meta }) => {
            setFile(file);
            setMeta(meta);
            setStep(2);
          }}
        />
      )}

      {/* STEP 2 */}
      {step === 2 && (
        <Step2Parameters
          meta={meta}
          onBack={() => setStep(1)}
          onNext={(params) => {
            setStep2(params);
            setStep(3);
          }}
        />
      )}

      {/* STEP 3 */}
      {step === 3 && (
        <Step3Preprocess
          file={file}
          meta={meta}
          step2={step2}
          onBack={() => setStep(2)}
          onAnalyzed={(data, fullParams) => {
            setResult({ data, params: fullParams });
            setStep(4);
          }}
        />
      )}

      {/* STEP 4 */}
      {step === 4 && result && (
        <Step4Decision
          file={file}
          step2={step2}
          result={result}
          onBack={() => setStep(3)}
          onContinue={(finalData, finalParams) => {
            setResult({ data: finalData, params: finalParams });
            setStep(5);
          }}
        />
      )}

      {/* STEP 5 */}
      {step === 5 && result && (
        <Step5Result
          result={result}
          onNew={() => resetAll()}
        />
      )}
    </div>
  );
}

/* (se este ProgressBar estiver neste mesmo arquivo) */
export function ProgressBar({ percent = 0 }) {
  return (
    <div className="w-full bg-gray-200 rounded">
      <div
        className="h-2 bg-green-500 transition-all"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
