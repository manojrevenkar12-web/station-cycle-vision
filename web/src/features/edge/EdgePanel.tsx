import { int, pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { EmptyState } from '../../shared/ui/StateBlock';
import styles from './edge.module.css';

export function EdgePanel({ r }: { r: Results }) {
  const e = r.edge;
  if (!e) return <EmptyState title="No edge results in this run">Run the edge step to export and time the models.</EmptyState>;
  const kb = (f: string) => `${Math.round((e.manifest.files[f] ?? 0) / 1024)} KB`;
  const agree = Math.min(...Object.values(e.live).map((v) => v.int8_float_agreement));
  return (
    <div style={{ display: 'grid', gap: 'var(--space-6)' }}>
      <div className={styles.facts}>
        <p className={styles.fact}>
          <b>{e.latency.int8.median_ms.toFixed(2)} ms</b>
          <span>per frame, one core, int8 (p99 {e.latency.int8.p99_ms.toFixed(2)} ms)</span>
        </p>
        <p className={styles.fact}>
          <b>{int(e.cameras_per_core_at_fps)}</b>
          <span>cameras one core keeps up with at {e.fps} fps (64 × 64)</span>
        </p>
        <p className={styles.fact}>
          <b>{pct(agree)}</b>
          <span>frames labelled the same by int8 and float, worst test set</span>
        </p>
        <p className={styles.fact}>
          <b>{pct(e.anonymisation_sample.leak_rate, 2)}</b>
          <span>person pixels left visible in a kept clip</span>
        </p>
      </div>
      <ol className={styles.chain} aria-label="How the edge path is checked">
        <li>
          <span>
            <b>Offline TCN → one-frame step with state</b>
            <br />
            <span>Ring buffer per dilated layer; zero state equals zero padding.</span>
          </span>
          <span className={styles.ok}>equal (test)</span>
        </li>
        <li>
          <span>
            <b>PyTorch → ONNX Runtime</b>
            <br />
            <span>
              Encoder {kb('encoder.onnx')}, streaming TCN {kb('tcn_step.onnx')}, detector {kb('detector.onnx')}.
            </span>
          </span>
          <span className={styles.ok}>&lt; 1e-3 (test)</span>
        </li>
        <li>
          <span>
            <b>Float encoder → int8 (static QDQ, per channel)</b>
            <br />
            <span>{kb('encoder.int8.onnx')}; calibrated on bench frames only.</span>
          </span>
          <span className={styles.ok}>{pct(agree)} same label</span>
        </li>
        <li>
          <span>
            <b>Live labels → events</b>
            <br />
            <span>The service emits exactly what the reference process emits, whatever the request chunking.</span>
          </span>
          <span className={styles.ok}>equal (test)</span>
        </li>
      </ol>
      <p style={{ color: 'var(--ink-soft)', maxWidth: 'var(--measure)', fontSize: 'var(--text-s)' }}>
        Timed on {e.cpu}, not on camera hardware; the frames are 64 × 64, so this compares implementations, not a deployment.
      </p>
    </div>
  );
}
