import serviceNir from '/images/service_nir.png';
import consultoria from '/images/consultoria_personalizada.png';
import treinamento from '/images/treinamentos_especializados.png';

export default function Servicos() {
  return (
    <section
      id="servicos"
      className="snap-start bg-gradient-to-r from-emerald-100 to-emerald-200 py-12 px-4"
    >
      <div className="mx-auto max-w-7xl space-y-6 reveal-from-left" data-delay="0.2">
        <h2 className="text-3xl md:text-4xl font-bold text-emerald-800 text-center">Serviços</h2>
        <ul className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          <li className="bg-white p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow">
            <a href="#analise" className="block space-y-3">
              <img src={serviceNir} alt="Análise & Processamento NIR" className="w-full rounded" />
              <h3 className="text-xl font-semibold">Análise & Processamento NIR</h3>
              <p className="text-sm text-gray-600">Transforme espectros NIR em insights acionáveis.</p>
            </a>
          </li>
          <li className="bg-white p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow">
            <a href="#consultoria" className="block space-y-3">
              <img src={consultoria} alt="Consultoria Personalizada NIR" className="w-full rounded" />
              <h3 className="text-xl font-semibold">Consultoria Personalizada</h3>
              <p className="text-sm text-gray-600">Soluções sob medida para integração e interpretação NIR.</p>
            </a>
          </li>
          <li className="bg-white p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow">
            <a href="#treinamento" className="block space-y-3">
              <img src={treinamento} alt="Treinamentos Especializados em NIR" className="w-full rounded" />
              <h3 className="text-xl font-semibold">Treinamentos Especializados</h3>
              <p className="text-sm text-gray-600">Capacitação prática do básico ao avançado em NIR.</p>
            </a>
          </li>
        </ul>
      </div>
    </section>
  );
}
