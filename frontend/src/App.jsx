import { Routes, Route, Navigate } from 'react-router-dom';
import Home from './Home.jsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/nir" element={<Navigate to="/nir_interface.html" replace />} />
    </Routes>
  );
}
