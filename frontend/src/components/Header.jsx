import { useState } from 'react';

export default function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="fixed inset-x-0 top-0 z-50 bg-white shadow-md">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
        <a href="/" className="flex items-center space-x-3">
          <img src="/images/Logo.jpg" alt="NIR CONNECT" className="h-14 md:h-16" />
          <span className="text-emerald-800 font-bold text-2xl md:text-3xl">NIR CONNECT</span>
        </a>
        <nav className="hidden md:flex space-x-8 text-lg">
          <a href="#sobre" className="hover:text-emerald-700">Sobre</a>
          <a href="#servicos" className="hover:text-emerald-700">Serviços</a>
          <a href="#contato" className="hover:text-emerald-700">Contato</a>
        </nav>
        <button id="btn-menu" className="md:hidden" aria-label="Toggle menu" onClick={() => setOpen(!open)}>
          <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
      {open && (
        <div id="mobile-menu" className="bg-white shadow-inner">
          <nav className="flex flex-col p-4 space-y-2">
            <a href="#sobre" className="hover:text-emerald-700">Sobre</a>
            <a href="#servicos" className="hover:text-emerald-700">Serviços</a>
            <a href="#contato" className="hover:text-emerald-700">Contato</a>
          </nav>
        </div>
      )}
    </header>
  );
}
