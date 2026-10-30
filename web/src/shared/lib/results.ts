// The shape of reports/results.json that the dashboard reads (written by `scv report`).
export interface SegProc {
  accuracy: number;
  edit: number;
  f1_10: number;
  f1_25: number;
  f1_50: number;
  cycle_recall: number;
  cycle_precision: number;
  cycle_time_mae_s: number | null;
  deviation_recall: number;
  deviation_precision: number;
  deviation_type_accuracy: number;
  cycles_true: number;
  cycles_found: number;
  cycles_matched: number;
  deviation_true: number;
  deviation_flagged: number;
  deviation_tp: number;
}
export interface Perception {
  hand_recall: number;
  hand_precision: number;
  mota: number;
  idf1: number;
  id_switches_per_min: number;
  leak_rate: number;
  over_redaction: number;
  frames_leaking_over_5px: number;
}
export interface CycleBox {
  start: number;
  end: number;
  deviations: string[];
  steps: string[];
}
export interface Timeline {
  station: string;
  fps: number;
  truth: number[];
  live: number[];
  offline: number[];
  cycles: CycleBox[];
  found: CycleBox[];
  standard_work: string[];
}
export interface Edge {
  latency: Record<'float' | 'int8', { median_ms: number; p99_ms: number; frames_per_s_per_core: number }>;
  cameras_per_core_at_fps: number;
  fps: number;
  debounce_frames: number;
  live: Record<string, SegProc & { int8_float_agreement: number }>;
  anonymisation_sample: { leak_rate: number; over_redaction: number };
  manifest: { files: Record<string, number> };
  cpu: string;
}
export interface Results {
  profile: string;
  generated_at: string;
  steps: string[];
  domains: string[];
  sizes: { train_frames: number; train_recordings: number; fps: number; test_recordings: Record<string, number> };
  model: { encoder_parameters: number; tcn_parameters: number; tcn_receptive_field_frames: number; detector_parameters: number };
  temporal: Record<string, Record<string, SegProc>>;
  onboarding: { zero_shot: SegProc; one_recording: SegProc };
  perception: Record<string, Perception>;
  timelines: Record<string, Timeline>;
  edge?: Edge;
  heldout_first_run?: { recorded_at: string; station_sha256: string; temporal: Record<string, SegProc> };
}

export function isResults(v: unknown): v is Results {
  const r = v as Results;
  return !!r && typeof r === 'object' && Array.isArray(r.domains) && !!r.temporal && !!r.timelines;
}
