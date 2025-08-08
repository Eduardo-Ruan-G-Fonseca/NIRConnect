import { useState } from 'react';
import { Link } from 'react-router-dom';
import logo from '/images/logo.svg';

export default function Header() {
  const [open, setOpen] = useState(false);

  return (
    <header className="bg-gradient-to-r from-green-700 to-blue-700 py-2 fixed top-0 w-full z-50">
      <nav className="container mx-auto flex items-center justify-between h-16">
        {/* logo à esquerda */}
        <a href="/" className="flex items-center">
          <img src={logo} alt="NIR CONNECT" className="h-14 md:h-16 object-contain" />
        </a>

        {/* links centrais */}
        <ul className="hidden md:flex space-x-8 flex-1 justify-center text-white">
          <li><a href="#sobre" className="hover:text-gray-200">Sobre</a></li>
          <li><a href="#servicos" className="hover:text-gray-200">Serviços</a></li>
          <li><a href="#contato" className="hover:text-gray-200">Contato</a></li>
        </ul>

        {/* botão à direita */}
        <Link
          to="/nir_interface.html"
          className="bg-white text-green-700 px-4 py-1 rounded hidden md:inline-block"
          reloadDocument
        >
          Acessar Plataforma
        </Link>

        {/* menu mobile */}
        <button
          id="btn-menu"
          className="md:hidden text-white"
          aria-label="Toggle menu"
          onClick={() => setOpen(!open)}
        >
          <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </nav>

      {open && (
        <div id="mobile-menu" className="md:hidden bg-gradient-to-r from-green-700 to-blue-700 px-6 py-4 text-white">
          <nav className="flex flex-col space-y-2">
            <a href="#sobre" className="hover:text-gray-200" onClick={() => setOpen(false)}>
              Sobre
            </a>
            <a href="#servicos" className="hover:text-gray-200" onClick={() => setOpen(false)}>
              Serviços
            </a>
            <a href="#contato" className="hover:text-gray-200" onClick={() => setOpen(false)}>
              Contato
            </a>
            <Link
              to="/nir_interface.html"
              className="bg-white text-green-700 px-4 py-1 rounded text-center"
              onClick={() => setOpen(false)}
              reloadDocument
            >
              Acessar Plataforma
            </Link>
          </nav>
        </div>
      )}
    </header>
  );
}
