export default function Hero() {
  return (
    <section id="hero" className="hero snap-start relative flex items-center justify-center text-center px-6 min-h-screen overflow-hidden">
      <div className="absolute inset-0 bg-black opacity-60"></div>
      <div className="relative max-w-3xl space-y-6">
        <h1 className="reveal-from-left text-7xl font-extrabold text-white" data-delay="0.1">
          NIR Web<br/>Inovadora & Eficiente
        </h1>
        <p className="reveal-from-left text-3xl text-gray-200" data-delay="0.3">
          Descubra como processar dados NIR de forma simples e moderna.
        </p>
      </div>
      <button
        className="absolute bottom-8 left-1/2 -translate-x-1/2 text-white text-4xl"
        onClick={() => window.scroll({ top: window.innerHeight, behavior: 'smooth' })}
      >
        ↓
      </button>
    </section>
  );
}
