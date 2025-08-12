import serviceNir from '/images/service_nir.png';

export default function Analise() {
  return (
    <section
      id="analise"
      className="snap-start bg-gradient-to-r from-sky-50 to-sky-100 py-24 px-6 detail-section"
    >
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <img
          src={serviceNir}
          alt="Serviço de Análise NIR"
          className="rounded-lg shadow-md reveal-from-left w-full h-auto object-cover"
          data-delay="0.2"
        />
        <div className="space-y-6 reveal-from-left" data-delay="0.4">
          <h2 className="text-5xl font-semibold text-emerald-800">
            Análise &amp; Processamento NIR
          </h2>
          <p className="text-xl leading-relaxed">
            Oferecemos análise avançada de espectros NIR com precisão e rapidez,
            otimizando controle de qualidade, pesquisa e processos industriais.
          </p>
          <a href="#contato" className="text-emerald-700 underline">
            Agende uma demonstração
          </a>
        </div>
      </div>
    </section>
  );
}
