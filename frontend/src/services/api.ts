import axios from "axios";

import type { ProjectDetail, ProjectSummary, SubmitResponse } from "./types";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? "",
  timeout: 15000,
});

export function apiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE ?? "";
  return base ? `${base.replace(/\/$/, "")}${path}` : path;
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const { data } = await api.get<ProjectSummary[]>("/api/projects");
  return data;
}

// `transcribeOnly` skips the Gemini pass: ingest → transcribe → export the transcription JSON.
export async function createProject(url: string, transcribeOnly = false): Promise<SubmitResponse> {
  const { data } = await api.post<SubmitResponse>("/api/projects", {
    url,
    transcribe_only: transcribeOnly,
  });
  return data;
}

export async function createProjectFromFile(
  file: File,
  transcribeOnly = false,
): Promise<SubmitResponse> {
  const form = new FormData();
  form.append("file", file);
  // No client timeout on the upload — a large local video can take a while to stream.
  const { data } = await api.post<SubmitResponse>(
    `/api/projects/upload?transcribe_only=${transcribeOnly}`,
    form,
    { timeout: 0 },
  );
  return data;
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
  const { data } = await api.get<ProjectDetail>(`/api/projects/${projectId}`);
  return data;
}

// Re-run Gemini + grounding on the cached transcript (no re-download/transcribe), overwriting the
// clip list. Returns a job to follow on the processing page.
export async function repromptProject(projectId: string): Promise<SubmitResponse> {
  const { data } = await api.post<SubmitResponse>(`/api/projects/${projectId}/reprompt`);
  return data;
}

// Judge the cached transcript, optionally restricted to a subset of the video's chapters (0-based
// indices into the start-sorted chapters; null = whole transcript). Resumes a run paused after
// transcribe for chapter selection. Returns a job to follow on the processing page.
export async function judgeProject(
  projectId: string,
  selectedChapters: number[] | null,
): Promise<SubmitResponse> {
  const { data } = await api.post<SubmitResponse>(`/api/projects/${projectId}/judge`, {
    selected_chapters: selectedChapters,
  });
  return data;
}

// Re-export the whole video as a DaVinci-ready .mov + transcription JSON to Downloads (no Gemini).
// Queued as a background job — follow it on the processing page. For no-Gemini projects.
export async function exportProject(projectId: string): Promise<SubmitResponse> {
  const { data } = await api.post<SubmitResponse>(`/api/projects/${projectId}/export`);
  return data;
}

// The exact judge prompt (instructions + numbered transcript) to paste into an external LLM.
export async function getProjectPrompt(projectId: string): Promise<string> {
  const { data } = await api.get<string>(`/api/projects/${projectId}/prompt`, {
    responseType: "text",
  });
  return data;
}

// Ground judge-format JSON (an external LLM's answer) and merge it into the clip list. Returns the
// updated project + clips and how many imported clips ended up in the list.
export async function importClips(
  projectId: string,
  clipsJson: string,
): Promise<ProjectDetail & { imported: number; total: number }> {
  const { data } = await api.post(`/api/projects/${projectId}/clips/import`, {
    clips_json: clipsJson,
  });
  return data;
}

export async function deleteProject(projectId: string): Promise<void> {
  await api.delete(`/api/projects/${projectId}`);
}

export function projectMediaUrl(projectId: string): string {
  return apiUrl(`/api/projects/${projectId}/media`);
}

export type AdjustParams = {
  edge: "start" | "end";
  unit: "word" | "sentence";
  direction: "expand" | "trim";
  count?: number;
};

export type ExportResult = {
  path: string;
  filename: string;
  transcript_path: string | null;
  transcript_filename: string | null;
};

// Render one clip to a DaVinci-ready .mov (DNxHR + PCM) plus a sidecar transcription JSON in the
// operator's Downloads folder, named after the clip's hook. Transcoding can take a few seconds, so
// there's no client timeout.
export async function exportClip(projectId: string, index: number): Promise<ExportResult> {
  const { data } = await api.post<ExportResult>(
    `/api/projects/${projectId}/clips/${index}/export`,
    {},
    { timeout: 0 },
  );
  return data;
}

// Nudge a clip's start/end by a word or a sentence (snapped to the transcript). Returns the
// updated project + full clip list.
export async function adjustClip(
  projectId: string,
  index: number,
  params: AdjustParams,
): Promise<ProjectDetail> {
  const { data } = await api.post<ProjectDetail>(
    `/api/projects/${projectId}/clips/${index}/adjust`,
    params,
  );
  return data;
}
