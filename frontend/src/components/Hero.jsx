import React from 'react';
import HeroImage from '/images/hero_image.png';

export default function Hero() {
  return (
    <section
      id="hero"
      className="hero relative flex items-center justify-center text-center px-6 min-h-screen overflow-hidden bg-cover bg-center"
      style={{ backgroundImage: `url(${HeroImage})` }}
    >
      <div className="absolute inset-0 bg-black opacity-60"></div>
      <div className="relative max-w-3xl space-y-6">
        <h1 className="reveal-from-left text-7xl font-extrabold text-white" data-delay="0.1">
          NIR Web<br />Inovadora & Eficiente
        </h1>
        <p className="reveal-from-left text-3xl text-gray-200" data-delay="0.3">
          Descubra como processar dados NIR de forma simples e moderna.
        </p>
      </div>

      <a href="#sobre" className="down-arrow-container" aria-label="Scroll down">
        <div className="down-arrow"></div>
      </a>
    </section>
  );
}
