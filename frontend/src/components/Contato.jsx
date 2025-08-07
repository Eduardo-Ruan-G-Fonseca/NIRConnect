import { useState } from 'react';

export default function Contato() {
  const [form, setForm] = useState({ name: '', email: '', phone: '', message: '', consent: false });

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setForm({ ...form, [name]: type === 'checkbox' ? checked : value });
  }

  function handleSubmit(e) {
    e.preventDefault();
    alert('Obrigado! Sua mensagem foi enviada.');
    setForm({ name: '', email: '', phone: '', message: '', consent: false });
  }

  return (
    <section
      id="contato"
      className="snap-start bg-gradient-to-r from-emerald-700 to-emerald-800 flex items-start min-h-screen py-32 px-6 detail-section"
    >
      <div className="max-w-4xl w-full mx-auto">
        <h2 className="reveal-from-left text-5xl font-bold text-white mb-8" data-delay="0.2">Fale Conosco</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <form id="contactForm" className="space-y-6 reveal-from-left" data-delay="0.4" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="name" className="block mb-1 font-medium text-white">Nome *</label>
              <input type="text" id="name" name="name" required value={form.name} onChange={handleChange} className="w-full border border-white rounded-md p-3 bg-transparent text-white focus:ring-2 focus:ring-white outline-none" />
            </div>
            <div>
              <label htmlFor="email" className="block mb-1 font-medium text-white">E-mail *</label>
              <input type="email" id="email" name="email" required value={form.email} onChange={handleChange} placeholder="tdiasflorestal@gmail.com" className="w-full border border-white rounded-md p-3 bg-transparent text-white focus:ring-2 focus:ring-white outline-none" />
            </div>
            <div>
              <label htmlFor="phone" className="block mb-1 font-medium text-white">Telefone *</label>
              <input type="tel" id="phone" name="phone" required value={form.phone} onChange={handleChange} className="w-full border border-white rounded-md p-3 bg-transparent text-white focus:ring-2 focus:ring-white outline-none" />
            </div>
            <div>
              <label htmlFor="message" className="block mb-1 font-medium text-white">Mensagem</label>
              <textarea id="message" name="message" rows="4" value={form.message} onChange={handleChange} className="w-full border border-white rounded-md p-3 bg-transparent text-white focus:ring-2 focus:ring-white outline-none"></textarea>
            </div>
            <div className="flex items-center space-x-2">
              <input type="checkbox" id="consent" name="consent" required checked={form.consent} onChange={handleChange} className="h-5 w-5 text-white" />
              <label htmlFor="consent" className="text-base text-white">Autorizo salvar meu contato *</label>
            </div>
            <button type="submit" className="w-full bg-white text-emerald-800 font-semibold py-4 rounded-md transition">ENVIAR</button>
          </form>
          <div className="bg-white bg-opacity-10 rounded-lg p-8 reveal-from-left" data-delay="0.6">
            <h3 className="text-3xl font-semibold text-white mb-6">Informações</h3>
            <p className="flex items-center text-white mb-4">
              <span className="text-2xl mr-2">📧</span>
              <a href="mailto:tdiasflorestal@gmail.com" className="underline">tdiasflorestal@gmail.com</a>
            </p>
            <p className="flex items-center text-white mb-4">
              <span className="text-2xl mr-2">📍</span>
              Perdões, MG – Brasil
            </p>
            <h4 className="text-xl font-semibold text-white mb-3">Horário de Atendimento</h4>
            <ul className="text-white space-y-1">
              <li>Seg – Sex: 9h – 22h</li>
              <li>Sáb: 9h – 18h</li>
              <li>Dom: 9h – 12h</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
