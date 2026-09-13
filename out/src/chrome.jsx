import { useEffect } from 'react';

// Ports the vanilla "tier-3 polish" behaviors (density toggle, keyboard
// shortcuts, mobile sidebar) from the static design's assets/app.js into a
// single mount-once React hook. Toasts are exposed on window so any page can
// call window.showToast(...) without importing anything extra.

function ensureToastHost() {
  let host = document.querySelector('.toast-host');
  if (!host) {
    host = document.createElement('div');
    host.className = 'toast-host';
    document.body.appendChild(host);
  }
  return host;
}

export function showToast(message, opts = {}) {
  const { type = 'info', duration = 3200 } = opts;
  const host = ensureToastHost();
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icons = { info: '●', success: '✓', warn: '⚠', error: '✕' };
  el.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span class="toast-msg"></span>`;
  el.querySelector('.toast-msg').textContent = message;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  const kill = () => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 220);
  };
  el.addEventListener('click', kill);
  setTimeout(kill, duration);
  return el;
}

export function useAppChrome(navItems, navigate) {
  // Density toggle (comfortable / compact), persisted.
  useEffect(() => {
    const KEY = 'apex-density';
    const apply = (mode) => document.documentElement.classList.toggle('density-compact', mode === 'compact');
    apply(localStorage.getItem(KEY) || 'comfortable');
    window.toggleDensity = () => {
      const next = document.documentElement.classList.contains('density-compact') ? 'comfortable' : 'compact';
      apply(next);
      localStorage.setItem(KEY, next);
      showToast(`Density: ${next}`, { type: 'info', duration: 1400 });
    };
    return () => { delete window.toggleDensity; };
  }, []);

  // Keyboard shortcuts: number keys jump to nav routes, D toggles density,
  // ? opens the help overlay, Esc closes overlays / mobile sidebar.
  useEffect(() => {
    function onKeyDown(e) {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || e.metaKey || e.ctrlKey || e.altKey) return;
      const overlay = document.querySelector('.shortcut-overlay');
      if (e.key === 'Escape') {
        overlay && overlay.classList.remove('show');
        document.body.classList.remove('sidebar-open');
        return;
      }
      if (e.key === '?') { overlay && overlay.classList.add('show'); return; }
      if (e.key.toLowerCase() === 'd') { window.toggleDensity && window.toggleDensity(); return; }
      const item = navItems.find((n) => n.key === e.key);
      if (item) navigate(item.to);
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [navItems, navigate]);
}

export function ShortcutOverlay({ navItems }) {
  return (
    <div className="shortcut-overlay" onClick={(e) => { if (e.target === e.currentTarget) e.currentTarget.classList.remove('show'); }}>
      <div className="shortcut-card glass">
        <div className="shortcut-head">
          <h3>Keyboard shortcuts</h3>
          <button
            type="button"
            className="btn btn-icon shortcut-close"
            onClick={(e) => e.currentTarget.closest('.shortcut-overlay').classList.remove('show')}
          >
            ✕
          </button>
        </div>
        <div className="shortcut-grid">
          {navItems.map((n) => (
            <div className="shortcut-row" key={n.key}>
              <kbd>{n.key}</kbd>
              <span>{n.label}</span>
            </div>
          ))}
          <div className="shortcut-row"><kbd>D</kbd><span>Toggle data density</span></div>
          <div className="shortcut-row"><kbd>?</kbd><span>Show this menu</span></div>
          <div className="shortcut-row"><kbd>Esc</kbd><span>Close overlays</span></div>
        </div>
      </div>
    </div>
  );
}
