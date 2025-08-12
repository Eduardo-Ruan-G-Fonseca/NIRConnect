import React from 'react';
import HeroImage from '/images/hero_image.png';

export default function Hero() {
  return (
    <section
      id="hero"
      className="hero relative flex items-center justify-center text-center px-6 min-h-screen overflow-hidden bg-cover bg-center"
      style={{ backgroundImage: `url(${HeroImage})` }}
    >
      {/* Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/90 via-black/60 to-black/10"></div>

      <div className="relative max-w-3xl space-y-6 px-4 mx-auto">
        <h1
          className="reveal-from-left text-6xl md:text-7xl font-extrabold text-white drop-shadow-lg"
          data-delay="0.1"
        >
          NIR Web<br />Inovadora &amp; Eficiente
        </h1>
        <p
          className="reveal-from-left text-xl md:text-3xl text-gray-200 drop-shadow-md"
          data-delay="0.3"
        >
          Descubra como processar dados NIR de forma simples e moderna.
        </p>

        <a
          href="#sobre"
          className="reveal-from-left inline-block bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 px-7 rounded-full shadow-xl transition-transform transform hover:scale-105"
          data-delay="0.5"
        >
          Saiba Mais
        </a>
      </div>

      {/* Seta */}
      <a href="#sobre" className="down-arrow-container" aria-label="Scroll down">
        <div className="down-arrow animate-bounce"></div>
      </a>
    </section>
  );
}
