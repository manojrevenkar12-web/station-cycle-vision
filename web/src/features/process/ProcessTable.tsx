import { pct, secs } from '../../shared/lib/format';
import type { Edge } from '../../shared/lib/results';
import { DOMAIN_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { EmptyState } from '../../shared/ui/StateBlock';
import t from '../../shared/ui/table.module.css';

/** What the camera box reports (int8, live, debounced), per test set. */
export function ProcessTable({ e }: { e: Edge | undefined }) {
  if (!e)
    return <EmptyState title="No edge results in this run">Run the edge step to see what the camera box reports.</EmptyState>;
  return (
    <ScrollRegion label="Process results by test set">
      <table className={t.table}>
        <caption>Live on the box: int8 encoder, streaming model, labels debounced over {e.debounce_frames} frames.</caption>
        <thead>
          <tr>
            <th scope="col">Test set</th>
            <th scope="col">Cycles found</th>
            <th scope="col">Cycle-time error</th>
            <th scope="col">Deviations caught</th>
            <th scope="col">Flags that were right</th>
            <th scope="col">Frame accuracy</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(e.live).map(([d, v]) => (
            <tr key={d}>
              <th scope="row" style={{ fontWeight: 500 }}>
                {DOMAIN_LABEL[d] ?? d}
              </th>
              <td>{pct(v.cycle_recall)}</td>
              <td>{secs(v.cycle_time_mae_s)}</td>
              <td>
                {pct(v.deviation_recall)}{' '}
                <span style={{ color: 'var(--ink-soft)' }}>
                  ({v.deviation_tp}/{v.deviation_true})
                </span>
              </td>
              <td>{pct(v.deviation_precision)}</td>
              <td>{pct(v.accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
