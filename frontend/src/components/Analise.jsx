export default function Analise() {
  return (
    <section
      id="analise"
      className="snap-start bg-gradient-to-r from-sky-50 to-sky-100 flex items-center min-h-screen py-32 px-6 detail-section"
    >
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <img
          src="/static/images/service_nir.png"
          alt="Serviço de Análise NIR"
          className="rounded-lg shadow-md reveal-from-left"
          data-delay="0.2"
        />
        <div className="space-y-6 reveal-from-left" data-delay="0.4">
          <h2 className="text-5xl font-semibold text-emerald-800">
            Análise & Processamento NIR
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
