// Theme constants used from TypeScript (charts, motion). Colours resolve CSS variables.
export const MOTION = {
  fast: 0.18,
  base: 0.32,
  slow: 0.7,
  ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
  sweepSeconds: 4,
} as const;

export const TIMELINE = {
  rowHeight: 22,
  rowGap: 10,
  cycleBand: 34,
  labelWidth: 96,
  width: 1100,
} as const;

export const STEP_LABEL: Record<string, string> = {
  idle: 'Idle',
  pick_base: 'Pick base',
  place_base: 'Place base',
  pick_screw: 'Pick screw',
  fasten: 'Fasten',
  inspect: 'Inspect',
  unload: 'Unload',
};

export const DEVIATION_LABEL: Record<string, string> = {
  missing: 'step missing',
  extra: 'extra step',
  order: 'out of order',
};

export const DOMAIN_LABEL: Record<string, string> = {
  train: 'Trained stations',
  camera: 'Camera moved',
  lighting: 'Lighting',
  busy: 'People walking past',
  new_station: 'New station',
  heldout: 'Held-out (blind)',
};

export const stepColor = (s: string) => `var(--s-${s})`;
