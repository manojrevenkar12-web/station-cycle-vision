import { pct, secs } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { EmptyState } from '../../shared/ui/StateBlock';
import t from '../../shared/ui/table.module.css';

export function HeldOut({ r }: { r: Results }) {
  const f = r.heldout_first_run;
  if (!f)
    return (
      <EmptyState title="No held-out first run yet">
        The blind station has not been run. Its first run is recorded once and never rewritten.
      </EmptyState>
    );
  return (
    <ScrollRegion label="Held-out station, first run">
      <table className={t.table}>
        <caption>
          First run {f.recorded_at.slice(0, 10)}; station file sha256 {f.station_sha256.slice(0, 12)}…; never rewritten.
        </caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Frame accuracy</th>
            <th scope="col">Edit</th>
            <th scope="col">Cycles found</th>
            <th scope="col">Cycle-time error</th>
            <th scope="col">Deviations caught</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(f.temporal).map(([m, v]) => (
            <tr key={m}>
              <th scope="row" style={{ fontWeight: m === 'causal-tcn' ? 700 : 500 }}>
                {m}
              </th>
              <td>{pct(v.accuracy)}</td>
              <td>{v.edit.toFixed(1)}</td>
              <td>{pct(v.cycle_recall)}</td>
              <td>{secs(v.cycle_time_mae_s)}</td>
              <td>{pct(v.deviation_recall)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
