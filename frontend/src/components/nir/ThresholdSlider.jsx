import React from "react";

export default function ThresholdSlider({ classes = [], valueMap = {}, onChange }) {
  const globalValue = classes.length
    ? classes.reduce((acc, c) => acc + (valueMap[c] ?? 0.5), 0) / classes.length
    : 0.5;

  const handleGlobal = (v) => {
    const updated = {};
    classes.forEach((c) => {
      updated[c] = v;
    });
    onChange && onChange(updated);
  };

  const handleClass = (cls, v) => {
    onChange && onChange({ ...valueMap, [cls]: v });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm mb-1">Threshold global: {globalValue.toFixed(2)}</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={globalValue}
          onChange={(e) => handleGlobal(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>
      {classes.map((cls) => (
        <div key={cls}>
          <label className="block text-sm mb-1">Classe {cls}: {(valueMap[cls] ?? globalValue).toFixed(2)}</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={valueMap[cls] ?? globalValue}
            onChange={(e) => handleClass(cls, parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      ))}
    </div>
  );
}
