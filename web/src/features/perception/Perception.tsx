import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { DOMAIN_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import t from '../../shared/ui/table.module.css';

export function Perception({ r }: { r: Results }) {
  return (
    <ScrollRegion label="Hand detection, tracking and anonymisation">
      <table className={t.table}>
        <caption>
          Hands found within 3 px; tracks scored with CLEAR MOT and IDF1; anonymisation against the true person pixels.
        </caption>
        <thead>
          <tr>
            <th scope="col">Test set</th>
            <th scope="col">Hands found</th>
            <th scope="col">Detections right</th>
            <th scope="col">MOTA</th>
            <th scope="col">IDF1</th>
            <th scope="col">ID switches / min</th>
            <th scope="col">Person pixels left visible</th>
            <th scope="col">Bench pixelated</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(r.perception).map(([d, v]) => (
            <tr key={d}>
              <th scope="row" style={{ fontWeight: 500 }}>
                {DOMAIN_LABEL[d] ?? d}
              </th>
              <td>{pct(v.hand_recall)}</td>
              <td>{pct(v.hand_precision)}</td>
              <td>{v.mota.toFixed(3)}</td>
              <td>{v.idf1.toFixed(3)}</td>
              <td>{v.id_switches_per_min.toFixed(1)}</td>
              <td>{pct(v.leak_rate, 2)}</td>
              <td>{pct(v.over_redaction)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
