import sobre from '/images/sobre.png';

export default function Sobre() {
  return (
    <section
      id="sobre"
      className="snap-start bg-gradient-to-r from-emerald-50 to-emerald-100 py-16 px-6"
    >
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <div className="space-y-8 reveal-from-left md:pl-8" data-delay="0.2">
          <h2 className="text-5xl font-bold text-emerald-800">Sobre</h2>
          <p className="text-xl leading-relaxed">
            Nossa plataforma NIR CONNECT entrega tecnologia web de ponta para análise NIR rápida e precisa.
          </p>
          <p className="text-xl leading-relaxed">
            Realize espectroscopia NIR diretamente no navegador com resultados em tempo real e relatórios completos que auxiliam sua tomada de decisão.
          </p>
        </div>
        <div className="reveal-from-left" data-delay="0.4">
          <img
            src={sobre}
            alt="Ilustração Sobre"
            className="w-full rounded-lg shadow-lg"
          />
        </div>
      </div>
    </section>
  );
}
