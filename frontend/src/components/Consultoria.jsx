import consultoria from '/images/consultoria_personalizada.png';

export default function Consultoria() {
  return (
    <section
      id="consultoria"
      className="snap-start bg-gradient-to-r from-sky-100 to-sky-200 flex items-center min-h-screen py-32 px-6 detail-section"
    >
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <img
          src={consultoria}
          alt="Consultoria Personalizada NIR"
          className="rounded-lg shadow-md reveal-from-left"
          data-delay="0.2"
        />
        <div className="space-y-6 reveal-from-left" data-delay="0.4">
          <h2 className="text-5xl font-semibold text-emerald-800">
            Consultoria Personalizada
          </h2>
          <p className="text-xl leading-relaxed">
            Integramos tecnologia NIR à sua operação: escolha de equipamentos,
            interpretação de dados e metodologias customizadas.
          </p>
          <a href="#contato" className="text-emerald-700 underline">
            Solicitar consultoria
          </a>
        </div>
      </div>
    </section>
  );
}
