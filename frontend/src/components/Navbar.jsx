import { useState } from "react";
import { Link } from "react-router-dom";

export default function Navbar({ voltarAoInicio, minimal: minimalProp }) {
  const [open, setOpen] = useState(false);

  // Detecta se está em /nir
  let rawPath = "/";
  if (typeof window !== "undefined") {
    const hash = window.location.hash ? window.location.hash.replace(/^#/, "") : "";
    rawPath = (hash || window.location.pathname || "/").toLowerCase();
  }
  const isNir = rawPath.includes("/nir");
  const minimal = typeof minimalProp === "boolean" ? minimalProp : isNir;

  return (
    <header className="bg-emerald-800 py-2 fixed top-0 w-full z-50 shadow-md">
      <nav
        className={`max-w-7xl mx-auto flex items-center h-16 px-4 md:px-12
          ${minimal ? "justify-center" : "justify-between"}`}
      >
        {/* Logo + Texto */}
        <Link
          to="/"
          onClick={voltarAoInicio}
          className="flex items-center ml-0 md:ml-4"
        >
          <img
            src="/images/Logo.png"
            alt="NIR CONNECT"
            className="h-14 md:h-16 object-contain logo-glow"
          />
          <span className="ml-2 text-white font-bold text-2xl md:text-3xl">
            NIR CONNECT
          </span>
        </Link>

        {/* Menu e botão só aparecem se NÃO for minimal */}
        {!minimal && (
          <>
            <ul className="hidden md:flex flex-1 justify-center space-x-8 text-lg text-white">
              <li><a href="#sobre" className="hover:text-emerald-200">Sobre</a></li>
              <li><a href="#servicos" className="hover:text-emerald-200">Serviços</a></li>
              <li><a href="#contato" className="hover:text-emerald-200">Contato</a></li>
            </ul>

            <Link
              to="/nir"
              className="hidden md:inline-block bg-emerald-700 hover:bg-emerald-800 text-white font-semibold px-6 py-2 rounded-md transition mr-0 md:mr-4"
            >
              Acessar Plataforma
            </Link>

            <button
              className="md:hidden text-white focus:outline-none"
              aria-label="Toggle menu"
              onClick={() => setOpen(!open)}
            >
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </>
        )}
      </nav>

      {!minimal && open && (
        <div className="md:hidden bg-emerald-800 px-6 py-4">
          <nav className="flex flex-col space-y-2 text-white">
            <a href="#sobre" className="hover:text-emerald-200" onClick={() => setOpen(false)}>Sobre</a>
            <a href="#servicos" className="hover:text-emerald-200" onClick={() => setOpen(false)}>Serviços</a>
            <a href="#contato" className="hover:text-emerald-200" onClick={() => setOpen(false)}>Contato</a>
            <Link
              to="/nir"
              onClick={() => setOpen(false)}
              className="bg-emerald-700 hover:bg-emerald-800 text-white px-6 py-2 rounded-md text-center mt-2"
            >
              Acessar Plataforma
            </Link>
          </nav>
        </div>
      )}
    </header>
  );
}
