import { Alert, Button, Progress, Tag } from "antd";
import { CircleAlert, CircleCheck, Clock3, FileJson, LoaderCircle, ListChecks } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { JobProgress, PipelineStage, StageProgress } from "../services/types";
import { ChapterPicker } from "./ChapterPicker";

const STAGES: Array<{ id: PipelineStage; label: string }> = [
  { id: "ingest", label: "Ingest" },
  { id: "transcribe", label: "Transcribe" },
  { id: "judge", label: "Judge" },
];

function stageIcon(stage: StageProgress) {
  if (stage.status === "done")
    return <CircleCheck aria-hidden size={18} className="text-emerald-600" />;
  if (stage.status === "error")
    return <CircleAlert aria-hidden size={18} className="text-red-600" />;
  if (stage.status === "running") {
    return <LoaderCircle aria-hidden size={18} className="animate-spin text-blue-600" />;
  }
  return <Clock3 aria-hidden size={18} className="text-slate-400" />;
}

function tagColor(status: StageProgress["status"]) {
  if (status === "done") return "green";
  if (status === "running") return "blue";
  if (status === "error") return "red";
  return "default";
}

function pct(stage: StageProgress) {
  if (stage.status === "done") return 100;
  if (stage.progress == null) return 0;
  return Math.round(stage.progress * 100);
}

export function PipelineProgress({ progress }: { progress: JobProgress | undefined }) {
  const navigate = useNavigate();
  const connection = progress?.connection ?? "connecting";
  const latest = progress?.latest;
  const ready = progress?.clipsReady;
  const awaiting = progress?.awaiting;
  const exportReady = progress?.exportReady;
  // No-Gemini runs (transcribe / re-export) reuse the third ("judge") row for export — relabel it.
  const mode = latest?.detail?.mode;
  const isExportMode = mode === "transcribe" || mode === "export";
  const errorEvent = latest?.status === "error" ? latest : null;

  return (
    <section className="surface p-4">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-normal text-slate-950">Processing</h1>
          <p className="text-sm text-slate-600">{latest?.message || "Waiting for progress"}</p>
        </div>
        <Tag
          color={connection === "error" ? "red" : connection === "open" ? "blue" : "default"}
          className="m-0 w-fit"
        >
          {connection}
        </Tag>
      </div>

      <div className="grid gap-3">
        {STAGES.map((item) => {
          const stage =
            progress?.stages[item.id] ??
            ({ stage: item.id, status: "idle", message: "", progress: null } as StageProgress);
          return (
            <div key={item.id} className="rounded-lg border border-slate-200 p-3">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  {stageIcon(stage)}
                  <span className="font-medium text-slate-900">
                    {item.id === "judge" && isExportMode ? "Export" : item.label}
                  </span>
                </div>
                <Tag color={tagColor(stage.status)} className="m-0 shrink-0">
                  {stage.status}
                </Tag>
              </div>
              <Progress
                percent={pct(stage)}
                size="small"
                status={stage.status === "error" ? "exception" : undefined}
              />
              <p className="mt-1 min-h-5 truncate text-sm text-slate-600">{stage.message}</p>
            </div>
          );
        })}
      </div>

      {awaiting ? (
        <ChapterPicker
          projectId={awaiting.projectId}
          chapters={awaiting.chapters}
          durationS={awaiting.durationS}
        />
      ) : null}

      {errorEvent ? (
        <Alert className="mt-4" type="error" showIcon message={errorEvent.message} />
      ) : null}

      {ready ? (
        <div className="mt-4 flex justify-end">
          <Button
            type="primary"
            icon={<ListChecks aria-hidden size={17} />}
            onClick={() => navigate(`/p/${ready.projectId}`)}
          >
            View {ready.nClips} clips
          </Button>
        </div>
      ) : null}

      {exportReady ? (
        <div className="mt-4 flex flex-col gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-start gap-2 text-sm text-emerald-800">
            <FileJson aria-hidden size={17} className="mt-0.5 shrink-0" />
            <div className="min-w-0">
              <p>Exported a DaVinci-ready video + transcript to your Downloads folder:</p>
              {exportReady.videoFilename ? (
                <p className="font-medium break-all">{exportReady.videoFilename}</p>
              ) : null}
              {exportReady.transcriptFilename ? (
                <p className="font-medium break-all">{exportReady.transcriptFilename}</p>
              ) : null}
            </div>
          </div>
          <Button
            className="shrink-0"
            icon={<ListChecks aria-hidden size={17} />}
            onClick={() => navigate(`/p/${exportReady.projectId}`)}
          >
            View project
          </Button>
        </div>
      ) : null}
    </section>
  );
}
