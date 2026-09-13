import React, { useEffect, useRef } from 'react';

// Readable multi-series line chart on canvas: smoothed curve, soft gradient
// fill under each line, larger labels, and a highlighted "now" point — built
// so a non-technical viewer can read the trend at a glance rather than
// parsing a wall of bars.
// series: { name: string, color: string, values: number[] }[]
// xLabels: string[] (same length as each series' values)
export default function LineChart({ series, xLabels, height = 220, yDomain, yFormat, fill = true }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !series || !series.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 500;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const nTicks = Math.max(...series.map((s) => s.values.length));
    const padL = 46, padR = 14, padT = 14, padB = 34;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    let yMin = Infinity, yMax = -Infinity;
    for (const s of series) for (const v of s.values) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    if (yDomain) { yMin = yDomain[0]; yMax = yDomain[1]; }
    if (!isFinite(yMin) || !isFinite(yMax)) { yMin = 0; yMax = 1; }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.12;
    yMin -= yPad; yMax += yPad;

    const xAt = (i) => padL + (plotW * i) / Math.max(1, nTicks - 1);
    const yAt = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // gridlines + y ticks — fewer, bolder, easier to scan
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '600 12px "Inter", sans-serif';
    ctx.textAlign = 'right';
    const ySteps = 4;
    for (let s = 0; s <= ySteps; s++) {
      const v = yMin + ((yMax - yMin) * s) / ySteps;
      const y = yAt(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.fillText(yFormat ? yFormat(v) : v.toFixed(1), padL - 8, y + 4);
    }

    // Smooth path helper (quadratic mid-point smoothing) so the trend reads
    // as one continuous curve rather than a jagged connect-the-dots line.
    function smoothPath(points) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i], p1 = points[i + 1];
        const midX = (p0.x + p1.x) / 2;
        const midY = (p0.y + p1.y) / 2;
        ctx.quadraticCurveTo(p0.x, p0.y, midX, midY);
      }
      const last = points[points.length - 1];
      ctx.lineTo(last.x, last.y);
    }

    for (const s of series) {
      const points = s.values.map((v, i) => ({ x: xAt(i), y: yAt(v) }));

      if (fill) {
        smoothPath(points);
        ctx.lineTo(points[points.length - 1].x, padT + plotH);
        ctx.lineTo(points[0].x, padT + plotH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
        grad.addColorStop(0, `${s.color}55`);
        grad.addColorStop(1, `${s.color}02`);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      smoothPath(points);
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.strokeStyle = s.color;
      ctx.shadowColor = `${s.color}80`;
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // highlighted dot at the last (current) point, with a soft halo
      const lastP = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(lastP.x, lastP.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = `${s.color}30`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lastP.x, lastP.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
    }

    // x-axis labels — rotated and truncated so long/dense labels never
    // overlap each other, however many points are plotted.
    if (xLabels && xLabels.length) {
      const step = Math.max(1, Math.ceil(nTicks / 6));
      const visibleCount = Math.ceil(nTicks / step);
      const slotW = (plotW / Math.max(1, visibleCount)) * 0.92;
      const longLabels = xLabels.some((l) => String(l).length > 8) || visibleCount > 5;
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.font = '600 11px "Inter", sans-serif';

      for (let i = 0; i < nTicks; i += step) {
        let label = String(xLabels[i]);
        // truncate with ellipsis to fit its slot
        while (ctx.measureText(label).width > (longLabels ? slotW * 1.6 : slotW) && label.length > 3) {
          label = `${label.slice(0, -2)}…`;
        }
        const x = xAt(i);
        if (longLabels) {
          ctx.save();
          ctx.textAlign = 'right';
          ctx.translate(x, height - 6);
          ctx.rotate(-Math.PI / 5);
          ctx.fillText(label, 0, 0);
          ctx.restore();
        } else {
          ctx.textAlign = 'center';
          ctx.fillText(label, x, height - 6);
        }
      }
    }
  }, [series, xLabels, height, yDomain, yFormat, fill]);

  return (
    <div className="mc-chart-wrap">
      <canvas ref={canvasRef} style={{ height }} />
    </div>
  );
}

export function LineLegend({ series }) {
  return (
    <div className="mc-legend">
      {series.map((s) => (
        <span key={s.name}>
          <i style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}
