// src/App.jsx
import { Routes, Route, Navigate } from "react-router-dom";
import Home from "./Home.jsx";
import NirPage from "./NirPage.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/nir" element={<NirPage />} />
      {/* opcional: redireciona quem cair no arquivo antigo */}
      <Route path="/nir_interface.html" element={<Navigate to="/nir" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
