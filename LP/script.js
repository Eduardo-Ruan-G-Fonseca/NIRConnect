// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navClose = document.getElementById('nav-close');
    const navLinks = document.querySelectorAll('.nav__link');
    const header = document.getElementById('header');

    // Mobile menu toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.add('show-menu');
        });
    }

    if (navClose) {
        navClose.addEventListener('click', () => {
            navMenu.classList.remove('show-menu');
        });
    }

    // Close menu when clicking on nav links
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('show-menu');
        });
    });

    // Header scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Generate spectrum lines
    const spectrum = document.getElementById('spectrum');
    if (spectrum) {
        for (let i = 0; i < 12; i++) {
            const line = document.createElement('div');
            line.className = 'spectrum-line';
            line.style.height = `${30 + Math.random() * 70}%`;
            line.style.animationDelay = `${i * 0.1}s`;
            spectrum.appendChild(line);
        }
    }

    // Generate particles
    const particles = document.getElementById('particles');
    if (particles) {
        for (let i = 0; i < 6; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${i * 0.5}s`;
            particles.appendChild(particle);
        }
    }

    // Testimonials slider
    const testimonialCards = document.querySelectorAll('.testimonial__card');
    const testimonialDots = document.querySelectorAll('.testimonial__dot');
    const prevBtn = document.getElementById('testimonial-prev');
    const nextBtn = document.getElementById('testimonial-next');
    let currentSlide = 0;

    function showSlide(index) {
        // Hide all cards
        testimonialCards.forEach(card => card.classList.remove('active'));
        testimonialDots.forEach(dot => dot.classList.remove('active'));

        // Show current card
        if (testimonialCards[index]) {
            testimonialCards[index].classList.add('active');
        }
        if (testimonialDots[index]) {
            testimonialDots[index].classList.add('active');
        }
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % testimonialCards.length;
        showSlide(currentSlide);
    }

    function prevSlide() {
        currentSlide = (currentSlide - 1 + testimonialCards.length) % testimonialCards.length;
        showSlide(currentSlide);
    }

    // Event listeners for testimonial navigation
    if (nextBtn) {
        nextBtn.addEventListener('click', nextSlide);
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', prevSlide);
    }

    // Dot navigation
    testimonialDots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            currentSlide = index;
            showSlide(currentSlide);
        });
    });

    // Auto-play testimonials
    setInterval(nextSlide, 5000);

    // Contact form handling
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                phone: formData.get('phone'),
                message: formData.get('message')
            };

            // Here you would typically send the data to your Python backend
            console.log('Form submitted:', data);
            
            // Show success message
            alert('Mensagem enviada com sucesso! Entraremos em contato em breve.');
            
            // Reset form
            this.reset();
        });
    }

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.about__item, .feature__card, .service__card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(40px)';
        el.style.transition = 'all 0.6s ease';
        observer.observe(el);
    });

    // Add staggered animation delays
    document.querySelectorAll('.about__item').forEach((item, index) => {
        item.style.transitionDelay = `${index * 0.1}s`;
    });

    document.querySelectorAll('.feature__card').forEach((card, index) => {
        card.style.transitionDelay = `${index * 0.1}s`;
    });

    document.querySelectorAll('.service__card').forEach((card, index) => {
        card.style.transitionDelay = `${index * 0.1}s`;
    });
});