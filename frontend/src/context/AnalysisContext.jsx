import { createContext, useContext, useState, useMemo } from "react";

const AnalysisContext = createContext(null);

export function AnalysisProvider({ children }) {
  const [step, setStep] = useState(0);       // 0..4
  const [file, setFile] = useState(null);
  const [targets, setTargets] = useState([]);
  const [meanSpectra, setMeanSpectra] = useState(null);
  const [spectraMatrix, setSpectraMatrix] = useState(null);

  // estados que usaremos nas próximas etapas:
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [selectedRanges, setSelectedRanges] = useState([]);
  const [selectedOpt, setSelectedOpt] = useState(null);

  const [lastParams, setLastParams] = useState(null);
  const [lastVip, setLastVip] = useState(null);
  const [lastYReal, setLastYReal] = useState(null);
  const [lastYPred, setLastYPred] = useState(null);
  const [lastTopVips, setLastTopVips] = useState(null);
  const [lastScores, setLastScores] = useState(null);
  const [lastInterpretacao, setLastInterpretacao] = useState(null);
  const [lastResumo, setLastResumo] = useState(null);

  const [preprocessOrder, setPreprocessOrder] = useState([]);

  const value = useMemo(() => ({
    // navegação
    step, setStep,
    // upload/colunas
    file, setFile,
    targets, setTargets,
    meanSpectra, setMeanSpectra,
    spectraMatrix, setSpectraMatrix,
    // próximos passos
    currentMetrics, setCurrentMetrics,
    selectedRanges, setSelectedRanges,
    selectedOpt, setSelectedOpt,
    lastParams, setLastParams,
    lastVip, setLastVip,
    lastYReal, setLastYReal,
    lastYPred, setLastYPred,
    lastTopVips, setLastTopVips,
    lastScores, setLastScores,
    lastInterpretacao, setLastInterpretacao,
    lastResumo, setLastResumo,
    preprocessOrder, setPreprocessOrder,
  }), [
    step, file, targets, meanSpectra, spectraMatrix,
    currentMetrics, selectedRanges, selectedOpt,
    lastParams, lastVip, lastYReal, lastYPred, lastTopVips, lastScores, lastInterpretacao, lastResumo,
    preprocessOrder
  ]);

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis deve ser usado dentro de <AnalysisProvider>");
  return ctx;
}
