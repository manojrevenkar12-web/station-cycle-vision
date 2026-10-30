export const pct = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}%`;
export const secs = (v: number | null) => (v === null || Number.isNaN(v) ? 'n/a' : `${v.toFixed(2)} s`);
export const pts = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(100 * v).toFixed(1)} pts`;
export const int = (v: number) => v.toLocaleString('en-GB');
export const us = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${Math.round(v)} µs`);
