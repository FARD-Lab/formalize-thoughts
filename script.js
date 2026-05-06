/* =========================================================
   Formalizing Latent Thoughts — interactive behaviour
   ========================================================= */

/* --- Theme toggle --- */
(function initTheme() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme')
      ?? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });
})();

/* --- Nav scroll style --- */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

/* =========================================================
   Hero canvas — flowing token particles
   ========================================================= */
(function initHeroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    canvas.height = canvas.offsetHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
  }
  resize();
  window.addEventListener('resize', resize, { passive: true });

  const w = () => canvas.offsetWidth;
  const h = () => canvas.offsetHeight;

  /* Central T node position */
  const cx = () => w() / 2;
  const cy = () => h() / 2;

  /* Particles */
  const PARTICLE_COUNT = 70;
  const particles = [];

  function makeParticle(i) {
    const side = Math.random() < 0.5 ? 'left' : 'right';
    const phase = (i / PARTICLE_COUNT) * Math.PI * 2;
    return {
      side,
      phase,
      t: Math.random(),
      speed: 0.0008 + Math.random() * 0.0008,
      spread: (Math.random() - 0.5) * 0.6,
      alpha: 0.2 + Math.random() * 0.5,
      radius: 1.5 + Math.random() * 2,
      color: Math.random() < 0.5 ? [129, 140, 248] : [77, 174, 126],
    };
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push(makeParticle(i));
  }

  function getPos(p, W, H) {
    const cx_ = W / 2;
    const cy_ = H / 2;
    const t = p.t;

    let x, y;
    if (p.side === 'left') {
      /* travel from left edge toward center */
      x = t * cx_;
      y = cy_ + p.spread * H * 0.35 * (1 - t);
    } else {
      /* travel from center toward right edge */
      x = cx_ + t * cx_;
      y = cy_ + p.spread * H * 0.35 * t;
    }
    return { x, y };
  }

  function draw() {
    const W = w(), H = h();
    ctx.clearRect(0, 0, W, H);

    /* Glow at center */
    const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, 120);
    grad.addColorStop(0,   'rgba(129, 140, 248, 0.12)');
    grad.addColorStop(0.5, 'rgba(129, 140, 248, 0.04)');
    grad.addColorStop(1,   'rgba(129, 140, 248, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    for (const p of particles) {
      p.t += p.speed;
      if (p.t > 1) {
        /* reset */
        p.t = 0;
        p.side = Math.random() < 0.5 ? 'left' : 'right';
        p.spread = (Math.random() - 0.5) * 0.6;
        p.speed = 0.0008 + Math.random() * 0.0008;
        p.alpha = 0.2 + Math.random() * 0.5;
        p.radius = 1.5 + Math.random() * 2;
        p.color = Math.random() < 0.5 ? [129, 140, 248] : [77, 174, 126];
      }

      const { x, y } = getPos(p, W, H);

      /* Fade near center (converge) */
      let alpha = p.alpha;
      const distToCenter = Math.abs(x - W/2);
      if (distToCenter < 60) {
        alpha *= distToCenter / 60;
      }

      const [r, g, b] = p.color;
      ctx.beginPath();
      ctx.arc(x, y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }
  draw();
})();

/* =========================================================
   Scroll-triggered fade-up animations
   ========================================================= */
(function initFadeUps() {
  const els = document.querySelectorAll('.fade-up');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  els.forEach((el, i) => {
    el.style.transitionDelay = `${(i % 4) * 0.08}s`;
    observer.observe(el);
  });
})();

/* =========================================================
   Scatter plot animation
   ========================================================= */
(function initScatter() {
  const svg = document.getElementById('scatter-svg');
  if (!svg) return;

  const candidates = document.getElementById('scatter-candidates');
  const oe         = document.getElementById('scatter-oe');
  const ideal      = document.getElementById('scatter-ideal');

  const observer = new IntersectionObserver((entries) => {
    if (!entries[0].isIntersecting) return;
    observer.disconnect();

    /* Step 1: candidates fade in */
    setTimeout(() => {
      candidates.style.transition = 'opacity 0.8s ease';
      candidates.style.opacity = '1';
    }, 200);

    /* Step 2: OE dot */
    setTimeout(() => {
      oe.style.transition = 'opacity 0.6s ease';
      oe.style.opacity = '1';
    }, 900);

    /* Step 3: ideal star */
    setTimeout(() => {
      ideal.style.transition = 'opacity 0.5s ease';
      ideal.style.opacity = '1';
    }, 1500);
  }, { threshold: 0.3 });

  observer.observe(svg);
})();

/* =========================================================
   Cite modal
   ========================================================= */
(function initCite() {
  const overlay  = document.getElementById('cite-overlay');
  const heroBtn  = document.getElementById('hero-cite-btn');
  const navBtn   = document.getElementById('nav-cite-btn');
  const closeBtn = document.getElementById('modal-close');
  const copyBtn  = document.getElementById('modal-copy');
  const footCopy = document.getElementById('copy-bib');

  function open()  { overlay.classList.add('open'); document.body.style.overflow = 'hidden'; }
  function close() { overlay.classList.remove('open'); document.body.style.overflow = ''; }

  if (heroBtn) heroBtn.addEventListener('click', open);
  if (navBtn)  navBtn.addEventListener('click', open);
  if (closeBtn) closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

  const bibText = document.getElementById('bib-text');

  function copyBib(btn) {
    const text = bibText ? bibText.textContent : '';
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1800);
    }).catch(() => {});
  }

  if (copyBtn)  copyBtn.addEventListener('click',  () => copyBib(copyBtn));
  if (footCopy) footCopy.addEventListener('click', () => copyBib(footCopy));
})();

/* =========================================================
   Hamburger / mobile nav
   ========================================================= */
(function initHamburger() {
  const btn  = document.getElementById('nav-hamburger');
  const menu = document.getElementById('nav-mobile-menu');
  if (!btn || !menu) return;

  function close() {
    menu.classList.remove('open');
    menu.setAttribute('aria-hidden', 'true');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-label', 'Open menu');
  }

  btn.addEventListener('click', () => {
    const opening = !menu.classList.contains('open');
    menu.classList.toggle('open');
    menu.setAttribute('aria-hidden', opening ? 'false' : 'true');
    btn.setAttribute('aria-expanded', opening ? 'true' : 'false');
    btn.setAttribute('aria-label', opening ? 'Close menu' : 'Open menu');
  });

  /* Close when a section link is tapped */
  menu.querySelectorAll('a').forEach(a => a.addEventListener('click', close));

  /* Close on Escape */
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

  /* Close when viewport grows past the mobile breakpoint */
  window.matchMedia('(min-width: 641px)').addEventListener('change', e => {
    if (e.matches) close();
  });
})();

/* =========================================================
   Axiom cards — keyboard accessible
   ========================================================= */
(function initAxiomCards() {
  document.querySelectorAll('.axiom-card').forEach(card => {
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        card.focus();
      }
    });
  });
})();
