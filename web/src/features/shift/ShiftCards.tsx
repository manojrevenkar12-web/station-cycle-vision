import { motion } from 'motion/react';
import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { DOMAIN_LABEL, MOTION } from '../../shared/theme/tokens';
import styles from './shift.module.css';

const MODELS = [
  { key: 'causal-tcn', label: 'with augmentation', color: 'var(--ink)' },
  { key: 'causal-tcn (no augmentation)', label: 'without', color: 'var(--s-place_base)' },
];
const METRICS = [
  { key: 'accuracy', label: 'Frame accuracy' },
  { key: 'cycle_recall', label: 'Cycles found' },
] as const;

/** One card per test set: does camera and light augmentation keep the model working? */
export function ShiftCards({ r }: { r: Results }) {
  return (
    <div style={{ display: 'grid', gap: 'var(--space-3)' }}>
      <p className={styles.key}>
        {MODELS.map((m) => (
          <span key={m.key}>
            <span className={styles.dot} style={{ background: m.color }} aria-hidden="true" />
            Causal TCN {m.label}
          </span>
        ))}
      </p>
      <div className={styles.grid}>
        {r.domains.map((d) => (
          <section key={d} className={styles.card} aria-label={DOMAIN_LABEL[d] ?? d}>
            <h3 className={styles.title}>{DOMAIN_LABEL[d] ?? d}</h3>
            {METRICS.map((m) => (
              <div key={m.key} className={styles.metric}>
                <span>{m.label}</span>
                <div className={styles.bars}>
                  {MODELS.map((mod) => {
                    const v = r.temporal[mod.key]?.[d]?.[m.key] ?? 0;
                    return (
                      <div key={mod.key} className={styles.bar}>
                        <div className={styles.track}>
                          <motion.div
                            className={styles.fill}
                            style={{ background: mod.color }}
                            initial={{ width: 0 }}
                            whileInView={{ width: `${100 * v}%` }}
                            viewport={{ once: true }}
                            transition={{ duration: MOTION.slow, ease: MOTION.ease }}
                          />
                        </div>
                        <span className="num">{pct(v)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </section>
        ))}
      </div>
    </div>
  );
}
