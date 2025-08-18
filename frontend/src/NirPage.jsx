import { useState } from "react";
import Navbar from "./components/Navbar.jsx";
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
  const [result, setResult] = useState(null);
  const [dataId, setDataId] = useState(null);

  function resetAll() {
    setStep(1);
    setFile(null);
    setMeta(null);
    setStep2(null);
    setResult(null);
    setDataId(null);
    window.scrollTo(0, 0);
  }

  return (
    <>
      <Navbar voltarAoInicio={resetAll} />

      <div className="pt-24 w-full max-w-[1400px] mx-auto px-4 md:px-6 lg:px-8 space-y-6">
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

        {/* Steps */}
        {step === 1 && (
          <Step1Upload
            onSuccess={({ file, meta }) => {
              setFile(file);
              setMeta(meta);
              setStep(2);
            }}
          />
        )}

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

        {step === 3 && (
          <Step3Preprocess
            file={file}
            meta={meta}
            step2={step2}
            onBack={() => setStep(2)}
            onAnalyzed={(data, fullParams) => {
              setResult({ data, params: fullParams });
              setDataId(data?.data_id);
              setStep(4);
            }}
          />
        )}

        {step === 4 && result && (
          <Step4Decision
            step2={step2}
            result={result}
            dataId={dataId}
            onBack={() => setStep(3)}
            onContinue={(finalData, finalParams) => {
              setResult({ data: finalData, params: finalParams });
              setStep(5);
            }}
          />
        )}

        {step === 5 && result && (
          <Step5Result
            result={result}
            onNew={() => resetAll()}
          />
        )}
      </div>
    </>
  );
}
