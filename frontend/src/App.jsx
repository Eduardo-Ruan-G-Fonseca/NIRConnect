import { Routes, Route } from 'react-router-dom';
import Home from './Home.jsx';
import NirPage from './NirPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/nir" element={<NirPage />} />
    </Routes>
  );
}
