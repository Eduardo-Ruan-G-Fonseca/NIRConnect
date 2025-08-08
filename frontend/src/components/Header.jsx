import { useState } from 'react';
import { Link } from 'react-router-dom';
import logo from '/images/logo.svg';

export default function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <nav className="bg-gradient-to-r from-green-700 to-blue-700 py-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6">
          <a href="/" className="flex items-center space-x-3">
            <img
              src={logo}
              alt="NIR CONNECT"
              className="h-14 md:h-16"
            />
            <span className="text-white font-bold text-2xl md:text-3xl">
              NIR CONNECT
            </span>
          </a>
          <div className="hidden md:flex flex-1 justify-center space-x-8 text-lg text-white">
            <a href="#sobre" className="hover:text-gray-200">
              Sobre
            </a>
            <a href="#servicos" className="hover:text-gray-200">
              Serviços
            </a>
            <a href="#contato" className="hover:text-gray-200">
              Contato
            </a>
          </div>
          <Link to="/nir_interface.html" className="btn hidden md:inline-block">
            Acessar a Plataforma
          </Link>
          <button
            id="btn-menu"
            className="md:hidden text-white"
            aria-label="Toggle menu"
            onClick={() => setOpen(!open)}
          >
            <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
        {open && (
          <div id="mobile-menu" className="md:hidden px-6 py-4 text-white">
            <nav className="flex flex-col space-y-2">
              <a href="#sobre" className="hover:text-gray-200">
                Sobre
              </a>
              <a href="#servicos" className="hover:text-gray-200">
                Serviços
              </a>
              <a href="#contato" className="hover:text-gray-200">
                Contato
              </a>
              <Link to="/nir_interface.html" className="btn">
                Acessar a Plataforma
              </Link>
            </nav>
          </div>
        )}
      </nav>
    </header>
  );
}
