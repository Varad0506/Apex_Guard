import React, { useEffect, useRef } from 'react';

// Draws N faint sample rollout paths + a bold mean line + a shaded min/max
// confidence band on a canvas. `series` is an array of {x, values[]} where
// values[] is one y-sample per path at that x — i.e. paths.length series
// stacked at each x tick. We accept the transposed shape instead: an array
// of paths, each an array of {x, y} points, plus a precomputed mean/band.
//
// Props:
//   paths: number[][]        — n_paths arrays of y-values, one per x tick
//   xLabels: (string|number)[] — one label per x tick, same length as each path
//   color: string             — CSS color for mean line / band / paths
//   height: number
//   yDomain: [number, number] — optional fixed y-axis range
export default function MonteCarloChart({ paths, xLabels, color = '#3dbfc9', height = 220, yDomain }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !paths || !paths.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 600;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const nTicks = paths[0].length;
    const padL = 8, padR = 8, padT = 10, padB = 20;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    let yMin = Infinity, yMax = -Infinity;
    for (const p of paths) for (const v of p) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    if (yDomain) { yMin = yDomain[0]; yMax = yDomain[1]; }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;

    const xAt = (i) => padL + (plotW * i) / Math.max(1, nTicks - 1);
    const yAt = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // min/max band
    const bandMin = [], bandMax = [], mean = [];
    for (let i = 0; i < nTicks; i++) {
      let mn = Infinity, mx = -Infinity, sum = 0;
      for (const p of paths) { const v = p[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
      bandMin.push(mn); bandMax.push(mx); mean.push(sum / paths.length);
    }

    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(bandMax[0]));
    for (let i = 1; i < nTicks; i++) ctx.lineTo(xAt(i), yAt(bandMax[i]));
    for (let i = nTicks - 1; i >= 0; i--) ctx.lineTo(xAt(i), yAt(bandMin[i]));
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.07)';
    ctx.fill();

    // faint sample paths (cap how many we draw for perf/legibility)
    const drawCount = Math.min(paths.length, 40);
    ctx.lineWidth = 1;
    ctx.strokeStyle = hexToRgba(color, 0.14);
    for (let p = 0; p < drawCount; p++) {
      ctx.beginPath();
      ctx.moveTo(xAt(0), yAt(paths[p][0]));
      for (let i = 1; i < nTicks; i++) ctx.lineTo(xAt(i), yAt(paths[p][i]));
      ctx.stroke();
    }

    // bold mean line
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(mean[0]));
    for (let i = 1; i < nTicks; i++) ctx.lineTo(xAt(i), yAt(mean[i]));
    ctx.stroke();

    // x-axis ticks
    if (xLabels && xLabels.length) {
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      const step = Math.max(1, Math.ceil(nTicks / 8));
      for (let i = 0; i < nTicks; i += step) {
        ctx.fillText(String(xLabels[i]), xAt(i), height - 5);
      }
    }
  }, [paths, xLabels, color, height, yDomain]);

  return (
    <div className="mc-chart-wrap">
      <canvas ref={canvasRef} style={{ height }} />
    </div>
  );
}

export function McLegend({ items }) {
  // items: {label, color, band?}[]
  return (
    <div className="mc-legend">
      {items.map((it) => (
        <span key={it.label}>
          <i className={it.band ? 'band' : ''} style={it.band ? undefined : { background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

function hexToRgba(hex, alpha) {
  if (hex.startsWith('rgba') || hex.startsWith('rgb')) return hex;
  const h = hex.replace('#', '');
  const bigint = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  const r = (bigint >> 16) & 255, g = (bigint >> 8) & 255, b = bigint & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}
