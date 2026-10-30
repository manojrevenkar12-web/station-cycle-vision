import { pct, secs } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import t from '../../shared/ui/table.module.css';

/** A station the models never saw, out of the box and after one labelled recording. */
export function Onboarding({ r }: { r: Results }) {
  const rows = [
    { name: 'Out of the box', v: r.onboarding.zero_shot },
    { name: 'After one labelled recording', v: r.onboarding.one_recording },
  ];
  return (
    <ScrollRegion label="Onboarding a new station">
      <table className={t.table}>
        <caption>
          New station S5: its own layout, three fastenings, inspection last. Only the temporal model is fine-tuned.
        </caption>
        <thead>
          <tr>
            <th scope="col">Labels from the new station</th>
            <th scope="col">Frame accuracy</th>
            <th scope="col">Edit</th>
            <th scope="col">Cycles found</th>
            <th scope="col">Cycle-time error</th>
            <th scope="col">Deviations caught</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ name, v }) => (
            <tr key={name}>
              <th scope="row" style={{ fontWeight: 500 }}>
                {name}
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
