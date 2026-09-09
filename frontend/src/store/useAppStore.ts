import { create } from "zustand";

import { deleteProject, listProjects } from "../services/api";
import type {
  JobProgress,
  JobRecord,
  PipelineStage,
  ProgressEvent,
  ProjectSummary,
} from "../services/types";

type LoadState = "idle" | "loading" | "ready" | "error";

const STAGES: PipelineStage[] = ["ingest", "transcribe", "judge"];

function createJobProgress(jobId: string): JobProgress {
  return {
    jobId,
    connection: "connecting",
    latest: null,
    clipsReady: null,
    awaiting: null,
    exportReady: null,
    stages: {
      ingest: { stage: "ingest", status: "idle", message: "", progress: null },
      transcribe: { stage: "transcribe", status: "idle", message: "", progress: null },
      judge: { stage: "judge", status: "idle", message: "", progress: null },
    },
  };
}

type AppState = {
  projects: ProjectSummary[];
  projectsState: LoadState;
  projectsError: string | null;
  jobs: Record<string, JobRecord>;
  jobProgress: Record<string, JobProgress>;
  fetchProjects: () => Promise<void>;
  removeProject: (projectId: string) => Promise<void>;
  upsertJob: (job: JobRecord) => void;
  setJobConnection: (jobId: string, connection: JobProgress["connection"]) => void;
  upsertJobEvent: (event: ProgressEvent) => void;
};

export const useAppStore = create<AppState>((set) => ({
  projects: [],
  projectsState: "idle",
  projectsError: null,
  jobs: {},
  jobProgress: {},
  async fetchProjects() {
    set({ projectsState: "loading", projectsError: null });
    try {
      const projects = await listProjects();
      set({ projects, projectsState: "ready" });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load projects";
      set({ projectsState: "error", projectsError: message });
    }
  },
  async removeProject(projectId) {
    await deleteProject(projectId);
    set((state) => ({ projects: state.projects.filter((p) => p.id !== projectId) }));
  },
  upsertJob(job) {
    set((state) => ({ jobs: { ...state.jobs, [job.id]: job } }));
  },
  setJobConnection(jobId, connection) {
    set((state) => {
      const current = state.jobProgress[jobId] ?? createJobProgress(jobId);
      return { jobProgress: { ...state.jobProgress, [jobId]: { ...current, connection } } };
    });
  },
  upsertJobEvent(event) {
    set((state) => {
      const stage = event.stage;
      const current = state.jobProgress[event.job_id] ?? createJobProgress(event.job_id);
      const nextStages = { ...current.stages };
      nextStages[stage] = {
        stage,
        status: event.status,
        message: event.message,
        progress: event.progress,
      };

      // A later stage going "running" — or the run pausing at "judge" for chapter selection —
      // implies the earlier stages finished. Mark them done so the three-row progress bar never
      // leaves a gap when an event is missed (e.g. only the last event is replayed on reconnect).
      const stageIndex = STAGES.indexOf(stage);
      if (event.status === "running" || event.detail.awaiting_selection === true) {
        for (const prior of STAGES.slice(0, stageIndex)) {
          if (nextStages[prior].status !== "done") {
            nextStages[prior] = { ...nextStages[prior], status: "done", progress: 1 };
          }
        }
      }

      const projectId = event.detail.project_id;
      const clipsReady =
        event.detail.clips_ready && typeof projectId === "string"
          ? {
              projectId,
              nClips: typeof event.detail.n_clips === "number" ? event.detail.n_clips : 0,
            }
          : current.clipsReady;
      const awaiting =
        event.detail.awaiting_selection && typeof projectId === "string"
          ? {
              projectId,
              chapters: Array.isArray(event.detail.chapters) ? event.detail.chapters : [],
              durationS:
                typeof event.detail.duration_s === "number" ? event.detail.duration_s : 0,
            }
          : current.awaiting;
      const exportReady =
        event.detail.export_ready && typeof projectId === "string"
          ? {
              projectId,
              videoFilename: event.detail.video_filename ?? "",
              transcriptFilename: event.detail.transcript_filename ?? "",
            }
          : current.exportReady;
      // "awaiting" is terminal for this job's stream too — no more events until a follow-up job.
      const status = event.detail.job_status;
      const terminal = status === "done" || status === "error" || status === "awaiting";

      return {
        jobProgress: {
          ...state.jobProgress,
          [event.job_id]: {
            ...current,
            connection: terminal ? "closed" : "open",
            stages: nextStages,
            latest: event,
            clipsReady,
            awaiting,
            exportReady,
          },
        },
      };
    });
  },
}));
