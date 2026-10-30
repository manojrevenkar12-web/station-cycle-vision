import { motion, useReducedMotion } from 'motion/react';
import { EdgePanel } from '../features/edge/EdgePanel';
import { HeldOut } from '../features/heldout/HeldOut';
import { Onboarding } from '../features/onboarding/Onboarding';
import { Perception } from '../features/perception/Perception';
import { ProcessTable } from '../features/process/ProcessTable';
import { ShiftCards } from '../features/shift/ShiftCards';
import { StepTimeline } from '../features/timeline/StepTimeline';
import { pct, secs } from '../shared/lib/format';
import type { Results } from '../shared/lib/results';
import { useResults } from '../shared/lib/useResults';
import { MOTION } from '../shared/theme/tokens';
import { Section } from '../shared/ui/Section';
import { Code, EmptyState, ErrorState, Skeleton } from '../shared/ui/StateBlock';
import styles from './app.module.css';

function Loaded({ r }: { r: Results }) {
  const reduce = useReducedMotion();
  const live = r.edge?.live.train;
  const rise = (i: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 10 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: MOTION.slow, ease: MOTION.ease, delay: 0.07 * i },
        };
  return (
    <>
      <section className={styles.hero} aria-labelledby="hero-h">
        <div className={styles.heroHead}>
          <motion.h1 id="hero-h" className={styles.headline} {...rise(0)}>
            Every cycle at the station, from the camera above it.
          </motion.h1>
          <motion.div style={{ display: 'grid', gap: 'var(--space-4)' }} {...rise(1)}>
            <p className={styles.intro}>
              Work steps recognised live on the camera box, turned into cycle times, and checked against the station's standard
              work. Workers are anonymised before any clip is kept. All numbers come from <Code>reports/results.json</Code>.
            </p>
            {live && (
              <div className={styles.facts}>
                <p className={styles.fact}>
                  <b>{pct(live.cycle_recall)}</b>
                  <span>cycles found on trained stations</span>
                </p>
                <p className={styles.fact}>
                  <b>{secs(live.cycle_time_mae_s)}</b>
                  <span>mean cycle-time error</span>
                </p>
                <p className={styles.fact}>
                  <b>{pct(live.deviation_recall)}</b>
                  <span>deviating cycles flagged</span>
                </p>
              </div>
            )}
          </motion.div>
        </div>
        <StepTimeline r={r} />
      </section>
      <Section
        id="process"
        kicker="On the box"
        title="Cycles and deviations, as the camera box reports them"
        lede="Int8, frame by frame, with the same debounce and cycle rules the service runs."
      >
        <ProcessTable e={r.edge} />
      </Section>
      <Section
        id="shift"
        kicker="Robustness"
        title="When the camera moves, the light changes or people walk past"
        lede="The labelled recordings all came from tidy benches. Augmenting the camera and light during training is the only defence here."
      >
        <ShiftCards r={r} />
      </Section>
      <Section
        id="onboarding"
        kicker="New station"
        title="How much labelling a new station costs"
        lede="Out of the box, then after one labelled two-minute recording."
      >
        <Onboarding r={r} />
      </Section>
      <Section
        id="perception"
        kicker="Hands and privacy"
        title="Hands tracked, people anonymised"
        lede="One small network finds hands and the tool and draws a person mask; the mask is pixelated before a clip leaves the box."
      >
        <Perception r={r} />
      </Section>
      <Section
        id="edge"
        kicker="Edge"
        title="One core, ONNX Runtime, no PyTorch"
        lede="The same models, exported and checked step by step."
      >
        <EdgePanel r={r} />
      </Section>
      <Section
        id="heldout"
        kicker="Held-out"
        title="A station written blind, run once"
        lede="Layout, product and camera written from a brief by someone who had not seen the code or the other stations."
      >
        <HeldOut r={r} />
      </Section>
      <footer className={styles.footer}>
        <p>
          Simulated stations: the video is rendered, not filmed. What transfers is the method, the checks and the plumbing; the
          accuracy figures do not.
        </p>
        <p>
          Run {r.generated_at.slice(0, 10)}, profile {r.profile}.
        </p>
      </footer>
    </>
  );
}

export function App() {
  const res = useResults();
  return (
    <main className={styles.page}>
      <header className={styles.masthead}>
        <span className={styles.brand}>station-cycle-vision</span>
        <span>Assembly-line cycle analytics</span>
      </header>
      {res.status === 'loading' && (
        <div className={styles.hero}>
          <Skeleton height={120} label="Loading results" />
          <Skeleton height={220} />
        </div>
      )}
      {res.status === 'error' && <ErrorState message={res.message} onRetry={res.retry} />}
      {res.status === 'empty' && (
        <EmptyState title="No results yet">
          Run <Code>scv all</Code> in the repository, then reload: the dashboard reads <Code>reports/results.json</Code>.
        </EmptyState>
      )}
      {res.status === 'ready' && <Loaded r={res.data} />}
    </main>
  );
}
