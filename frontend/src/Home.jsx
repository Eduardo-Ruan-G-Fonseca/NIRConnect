import { useEffect } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import Sobre from './components/Sobre';
import Servicos from './components/Servicos';
import Analise from './components/Analise';
import Consultoria from './components/Consultoria';
import Treinamento from './components/Treinamento';
import Contato from './components/Contato';

export default function Home() {
  useEffect(() => {
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach((a) => {
      a.addEventListener('click', (e) => {
        if (a.getAttribute('href').startsWith('#')) {
          e.preventDefault();
          const tgt = document.querySelector(a.getAttribute('href'));
          if (tgt) {
            const top = tgt.getBoundingClientRect().top + window.pageYOffset - 80;
            window.scrollTo({ top, behavior: 'smooth' });
          }
        }
      });
    });

    // Slide-in reveals
    document.querySelectorAll('.reveal-from-left').forEach((el) => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              el.style.animationDelay = el.dataset.delay + 's';
              el.classList.add('animate-from-left');
            } else {
              el.classList.remove('animate-from-left');
            }
          });
        },
        { threshold: 0.2 }
      );
      observer.observe(el);
    });

    // Fade-in-up for detail sections
    document.querySelectorAll('.detail-section').forEach((sec) => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            entry.target.classList.toggle('visible', entry.isIntersecting);
          });
        },
        { threshold: 0.2 }
      );
      observer.observe(sec);
    });
  }, []);

  return (
    <>
      <Header />
      <Hero />
      <Sobre />
      <Servicos />
      <Analise />
      <Consultoria />
      <Treinamento />
      <Contato />
    </>
  );
}
