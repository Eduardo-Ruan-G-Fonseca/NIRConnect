import { useState, useRef } from "react";
import { setDatasetId } from "../../api/http";

export default function Step1Upload({ onSuccess }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const API_BASE =
    window.API_BASE || `${location.protocol}//${location.hostname}:8000`;

  function handleFileChange(e) {
    const f = e.target.files?.[0] || null;
    setSelectedFile(f);
    setError("");
    setStatus("");
    setProgress(0);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setStatus("");

    const file = selectedFile || inputRef.current?.files?.[0];
    if (!file) {
      setError("Selecione um arquivo .csv ou .xlsx.");
      return;
    }

    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx"].includes(ext)) {
      setError("Arquivo deve ser .csv ou .xlsx.");
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError("Arquivo excede 100MB.");
      return;
    }

    setLoading(true);
    setProgress(0);
    setStatus("Enviando arquivo...");

    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);

    xhr.responseType = "json";

    xhr.upload.addEventListener("progress", (ev) => {
      if (ev.lengthComputable) {
        setProgress(Math.round((ev.loaded / ev.total) * 100));
      }
    });

    xhr.onload = () => {
      setLoading(false);
      if (xhr.status === 200) {
        setProgress(100);
        setStatus("Arquivo enviado com sucesso!");
        const meta = xhr.response ?? safeParse(xhr.responseText);
        if (!meta) {
          setError("Não foi possível interpretar a resposta do servidor.");
          return;
        }
        if (meta?.dataset_id) {
          localStorage.setItem("nir.datasetId", meta.dataset_id);
          localStorage.setItem("nir.columns", JSON.stringify(meta.columns || []));
          localStorage.setItem("nir.targets", JSON.stringify(meta.targets || []));
          setDatasetId(meta.dataset_id);
        }
        onSuccess({ file, meta });
      } else {
        const msg =
          xhr.response?.detail ||
          xhr.response?.message ||
          xhr.statusText ||
          "Falha no upload.";
        setError(String(msg));
        setStatus("");
      }
    };

    xhr.onerror = () => {
      setLoading(false);
      setError("Erro de rede ao enviar arquivo.");
      setStatus("");
    };

    xhr.open("POST", `${API_BASE}/columns`);
    xhr.send(fd);
  }

  function safeParse(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h2 className="text-lg font-semibold">1. Upload</h2>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Upload de Arquivo (.csv ou .xlsx)
          </label>
          <input
            id="nir-file"
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx"
            className="mt-1 block w-full border rounded p-2"
            onChange={handleFileChange}
          />
        </div>

        {/* Preview do arquivo escolhido */}
        {selectedFile && (
          <div className="text-sm bg-gray-50 border rounded p-2">
            <div>
              <strong>Arquivo:</strong> {selectedFile.name}
            </div>
            <div>
              <strong>Tamanho:</strong>{" "}
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </div>
            <div>
              <strong>Tipo:</strong> {selectedFile.type || "desconhecido"}
            </div>
          </div>
        )}

        {/* Barra de progresso + status */}
        {loading && (
          <>
            <div className="w-full bg-gray-200 rounded">
              <div
                className="h-2 bg-green-500 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-sm text-gray-600">
              {progress}% — {status}
            </div>
          </>
        )}

        {/* Mensagens */}
        {status && !loading && (
          <div className="text-sm text-green-700">{status}</div>
        )}
        {error && <div className="text-sm text-red-600">{error}</div>}

        <button
          type="submit"
          className="bg-[#2e5339] hover:bg-[#305e6b] text-white py-2 px-4 rounded disabled:opacity-60"
          disabled={loading || !selectedFile}
        >
          Avançar
        </button>
      </form>
    </div>
  );
}
