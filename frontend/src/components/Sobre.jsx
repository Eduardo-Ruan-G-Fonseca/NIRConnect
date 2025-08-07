export default function Sobre() {
  return (
    <section
      id="sobre"
      className="snap-start bg-gradient-to-r from-emerald-50 to-emerald-100 flex flex-col md:flex-row md:space-x-24 items-center min-h-screen py-32 px-6 detail-section"
    >
      <div className="md:w-5/12 max-w-4xl space-y-8 reveal-from-left md:pl-16" data-delay="0.2">
        <h2 className="text-5xl font-bold text-emerald-800">Sobre</h2>
        <p className="text-xl leading-relaxed">
          Nossa plataforma NIR CONNECT entrega tecnologia web de ponta para análise NIR rápida e precisa.
        </p>
        <p className="text-xl leading-relaxed">
          Realize espectroscopia NIR diretamente no navegador com resultados em tempo real e relatórios completos que auxiliam sua tomada de decisão.
        </p>
      </div>
      <div className="md:w-7/12 mt-12 md:mt-0 flex justify-end reveal-from-left" data-delay="0.4">
        <img src="/images/sobre.png" alt="Ilustração Sobre" className="w-full rounded-lg shadow-lg" />
      </div>
    </section>
  );
}
