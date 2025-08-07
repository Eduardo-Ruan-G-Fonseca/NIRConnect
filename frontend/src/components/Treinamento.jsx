export default function Treinamento() {
  return (
    <section
      id="treinamento"
      className="snap-start bg-gradient-to-r from-sky-50 to-sky-100 flex items-center min-h-screen py-32 px-6 detail-section"
    >
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <img
          src="/images/treinamentos_especializados.png"
          alt=""
          className="rounded-lg shadow-md reveal-from-left"
          data-delay="0.2"
        />
        <div className="space-y-6 reveal-from-left" data-delay="0.4">
          <h2 className="text-5xl font-semibold text-emerald-800">Treinamentos Especializados</h2>
          <p className="text-xl leading-relaxed">
            Cursos completos de espectroscopia NIR, do fundamental ao avançado, para elevar a expertise da sua equipe.
          </p>
          <a href="#contato" className="text-emerald-700 underline">Inscrever equipe</a>
        </div>
      </div>
    </section>
  );
}
