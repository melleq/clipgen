// The scoring axes Gemini returns per clip; also the display/column order. hook + payoff +
// virality gate the clip (all three must be strong); the rest modulate.
export const SCORE_DIMENSIONS = [
  "hook",
  "payoff",
  "virality",
  "emotion",
  "quotability",
  "controversy",
  "standalone",
] as const;

export type ProjectStatus =
  | "queued"
  | "ingesting"
  | "transcribing"
  | "awaiting" // transcribed; paused for the operator to pick which chapters to send to Gemini
  | "judging"
  | "done"
  | "error";

// One chapter of the source video (its own table of contents), used as a segmentation hint and as
// the selectable unit the operator picks from before sending the transcript to Gemini.
export type Chapter = {
  start_s: number;
  title: string;
};

// One project = one ingested video and its clips (the `meta.json` row + derived counts).
export type ProjectSummary = {
  id: string;
  title: string;
  source_url: string;
  duration_s: number;
  status: ProjectStatus;
  error: string;
  created_at: number;
  n_clips?: number;
  has_video?: boolean;
};

// One grounded clip (matches the backend clipgen.clips.ClipItem shape). `segments` are the clip's
// spoken spans in seconds — one for a simple clip, several for a compound (spliced) clip, with the
// gaps between them being the filler that was cut. `duration_s` is spoken time (sum of segments).
export type ClipItem = {
  start_s: number;
  end_s: number;
  duration_s: number;
  hook: string;
  hook_sentence: string;
  // Where the payoff lands (verbatim line), what the viewer gets, and who shares it and why —
  // the judge must justify every clip; grounding verifies the payoff line is inside the clip.
  payoff_sentence?: string;
  payoff?: string;
  share_trigger?: string;
  reason: string;
  text: string;
  scores: Record<string, number>;
  content_score: number;
  clip_type: "simple" | "compound";
  segments: [number, number][];
  splice_reason: string;
  source?: "gemini" | "imported";
};

export type ProjectDetail = {
  project: ProjectSummary;
  clips: ClipItem[];
};

export type JobRecord = {
  id: string;
  project_id: string;
  source_kind: "url" | "file";
  status: "queued" | "running" | "awaiting" | "done" | "error";
  error: string | null;
  created_at: number;
  updated_at: number;
};

export type SubmitResponse = {
  project_id: string;
  job_id: string;
  job: JobRecord;
};

export type PipelineStage = "ingest" | "transcribe" | "judge";
export type ProgressStatus = "queued" | "running" | "done" | "error";

export type ProgressEvent = {
  job_id: string;
  stage: PipelineStage;
  status: ProgressStatus;
  message: string;
  progress: number | null;
  detail: {
    project_id?: string;
    // Which pipeline the job is running. "transcribe"/"export" are the no-Gemini export paths.
    mode?: "full" | "judge" | "transcribe" | "export";
    n_clips?: number;
    clips_ready?: boolean;
    // Set when a full run pauses after transcribe: the source's chapters to choose from.
    awaiting_selection?: boolean;
    chapters?: Chapter[];
    duration_s?: number;
    // Set when a no-Gemini run (transcribe/export) finishes: the exported video + transcription.
    export_ready?: boolean;
    video_filename?: string;
    video_path?: string;
    transcript_filename?: string;
    transcript_path?: string | null;
    job_status?: "done" | "error" | "awaiting";
    [key: string]: unknown;
  };
};

export type StageProgress = {
  stage: PipelineStage;
  status: ProgressStatus | "idle";
  message: string;
  progress: number | null;
};

export type JobProgress = {
  jobId: string;
  connection: "connecting" | "open" | "closed" | "error";
  stages: Record<PipelineStage, StageProgress>;
  latest: ProgressEvent | null;
  clipsReady: { projectId: string; nClips: number } | null;
  // Set when the run paused after transcribe for the operator to pick chapters to send to Gemini.
  awaiting: { projectId: string; chapters: Chapter[]; durationS: number } | null;
  // Set when a no-Gemini run finished and wrote the video + transcription JSON to Downloads.
  exportReady: { projectId: string; videoFilename: string; transcriptFilename: string } | null;
};
