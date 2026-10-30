import { motion, useReducedMotion } from 'motion/react';
import { useState } from 'react';
import { secs } from '../../shared/lib/format';
import type { Results, Timeline } from '../../shared/lib/results';
import { DEVIATION_LABEL, DOMAIN_LABEL, MOTION, STEP_LABEL, stepColor, TIMELINE } from '../../shared/theme/tokens';
import { Segmented } from '../../shared/ui/Segmented';
import styles from './timeline.module.css';

function runs(labels: number[]): { step: number; start: number; end: number }[] {
  const out: { step: number; start: number; end: number }[] = [];
  labels.forEach((l, i) => {
    const last = out[out.length - 1];
    if (last && last.step === l) last.end = i + 1;
    else out.push({ step: l, start: i, end: i + 1 });
  });
  return out;
}

function Row({ labels, steps, y, x }: { labels: number[]; steps: string[]; y: number; x: (f: number) => number }) {
  return (
    <g>
      {runs(labels).map((r) => (
        <rect
          key={r.start}
          x={x(r.start)}
          y={y}
          width={Math.max(0.5, x(r.end) - x(r.start))}
          height={TIMELINE.rowHeight}
          fill={stepColor(steps[r.step] ?? 'idle')}
        />
      ))}
    </g>
  );
}

/** Two minutes of one station: what the box reported live, the offline re-segmentation, and the truth. */
export function StepTimeline({ r }: { r: Results }) {
  const domains = Object.keys(r.timelines);
  const [domain, setDomain] = useState(domains[0] ?? 'train');
  const reduce = useReducedMotion();
  const t: Timeline = r.timelines[domain]!;
  const n = t.truth.length;
  const { labelWidth: L, width: W, rowHeight: RH, rowGap: G, cycleBand: CB } = TIMELINE;
  const x = (f: number) => L + (f / n) * (W - L - 8);
  const rows = [
    { key: 'live', name: 'On the box', labels: t.live },
    { key: 'offline', name: 'Offline', labels: t.offline },
    { key: 'truth', name: 'What happened', labels: t.truth },
  ];
  const H = CB + rows.length * (RH + G) + 28;
  const rowY = (i: number) => CB + i * (RH + G);
  const truthDev = t.cycles.filter((c) => c.deviations.length);
  const foundDev = t.found.filter((c) => c.deviations.length);
  const ticks = Array.from({ length: Math.floor(n / t.fps / 20) + 1 }, (_, i) => i * 20);
  const summary = `${DOMAIN_LABEL[domain] ?? domain}, station ${t.station}: ${t.cycles.length} cycles happened, ${truthDev.length} deviated; the box reported ${t.found.length} cycles and flagged ${foundDev.length}.`;
  return (
    <figure className={styles.wrap} style={{ margin: 0 }}>
      <Segmented label="Recording" options={domains} value={domain} onChange={setDomain} format={(d) => DOMAIN_LABEL[d] ?? d} />
      {/* biome-ignore lint/a11y/noNoninteractiveTabindex: the board scrolls sideways on a phone; keyboard users must reach it */}
      <section className={styles.board} tabIndex={0} aria-label="Step timeline, scrolls sideways">
        <svg viewBox={`0 0 ${W} ${H}`} className={styles.svg} role="img" aria-label={summary}>
          <defs>
            <clipPath id={`sweep-${domain}`}>
              <motion.rect
                key={domain}
                x={0}
                y={0}
                height={H}
                initial={{ width: reduce ? W : L }}
                animate={{ width: W }}
                transition={{ duration: reduce ? 0 : MOTION.sweepSeconds, ease: 'linear' }}
              />
            </clipPath>
          </defs>
          {rows.map((row, i) => (
            <text key={row.key} x={0} y={rowY(i) + RH / 2 + 4} className={styles.rowLabel}>
              {row.name}
            </text>
          ))}
          <g clipPath={`url(#sweep-${domain})`}>
            {rows.map((row, i) => (
              <Row key={row.key} labels={row.labels} steps={r.steps} y={rowY(i)} x={x} />
            ))}
            {t.found.map((c) => {
              const x0 = x(c.start * t.fps);
              const x1 = x(c.end * t.fps);
              const dev = c.deviations.length > 0;
              return (
                <g key={`f${c.start}`}>
                  <path
                    d={`M${x0},${CB - 6} V${CB - 12} H${x1} V${CB - 6}`}
                    fill="none"
                    stroke={dev ? 'var(--deviation)' : 'var(--ink)'}
                    strokeWidth={1.5}
                  />
                  <text x={(x0 + x1) / 2} y={CB - 16} textAnchor="middle" className={dev ? styles.devLabel : styles.cycleLabel}>
                    {dev ? c.deviations.map((d) => DEVIATION_LABEL[d] ?? d).join(', ') : secs(c.end - c.start)}
                  </text>
                </g>
              );
            })}
            {truthDev.map((c) => (
              <rect
                key={`t${c.start}`}
                x={x(c.start * t.fps)}
                y={rowY(2) - 3}
                width={x(c.end * t.fps) - x(c.start * t.fps)}
                height={RH + 6}
                fill="none"
                stroke="var(--deviation)"
                strokeWidth={2}
                strokeDasharray="4 3"
              />
            ))}
          </g>
          {ticks.map((s) => (
            <text key={s} x={x(s * t.fps)} y={H - 6} className={styles.axis} textAnchor="middle">
              {s}s
            </text>
          ))}
        </svg>
      </section>
      <ul className={styles.legend}>
        {r.steps.map((s) => (
          <li key={s}>
            <span className={styles.swatch} style={{ background: stepColor(s) }} aria-hidden="true" />
            {STEP_LABEL[s] ?? s}
          </li>
        ))}
        <li>
          <span className={styles.swatch} style={{ border: '2px dashed var(--deviation)' }} aria-hidden="true" />
          cycle that really deviated
        </li>
      </ul>
      <figcaption className={styles.caption}>
        {summary} Brackets above are the cycles the box reported, labelled with the cycle time, or in orange with what differed
        from standard work ({t.standard_work.length} steps).
      </figcaption>
    </figure>
  );
}
