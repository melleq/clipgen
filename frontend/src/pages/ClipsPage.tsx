import { Alert, Button, Empty, Input, Modal, Spin, Tag, message } from "antd";
import { ArrowLeft, ClipboardCopy, FilePlus2, FileVideo, Scissors, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ClipCard } from "../components/ClipCard";
import {
  adjustClip,
  exportClip,
  exportProject,
  getProject,
  getProjectPrompt,
  importClips,
  projectMediaUrl,
  repromptProject,
} from "../services/api";
import type { AdjustParams } from "../services/api";
import type { ClipItem, ProjectDetail } from "../services/types";

type LoadState = "loading" | "ready" | "error";

const clipKey = (c: ClipItem) => `${c.start_s}-${c.end_s}`;

export function ClipsPage() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [reprompting, setReprompting] = useState(false);
  const [exportingProject, setExportingProject] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState("");
  const [importing, setImporting] = useState(false);
  // Which clip is loaded in the shared player (for the highlight + "now playing" caption).
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  // Active preview: the clip's spoken segments and which one is playing. The timeupdate handler
  // jumps over the spliced-out gaps and pauses at the end, so playback stays inside the clip.
  const segmentsRef = useRef<[number, number][] | null>(null);
  const segIndexRef = useRef(0);

  useEffect(() => {
    if (!projectId) return;
    let alive = true;
    setState("loading");
    getProject(projectId)
      .then((data) => {
        if (!alive) return;
        setDetail(data);
        setState("ready");
      })
      .catch((err) => {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "Failed to load project");
        setState("error");
      });
    return () => {
      alive = false;
    };
  }, [projectId]);

  // One bounded-playback controller for the shared player, attached once.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    function onTimeUpdate() {
      const v = videoRef.current;
      const segs = segmentsRef.current;
      if (!v || !segs) return;
      const end = segs[segIndexRef.current][1];
      if (v.currentTime >= end) {
        if (segIndexRef.current + 1 < segs.length) {
          segIndexRef.current += 1;
          v.currentTime = segs[segIndexRef.current][0]; // skip the spliced-out gap
        } else {
          v.pause();
          segmentsRef.current = null; // clip finished — release the bound
          setActiveKey(null);
        }
      }
    }
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [detail]);

  const preview = useCallback((clip: ClipItem) => {
    const video = videoRef.current;
    if (!video || !clip.segments.length) return;
    segmentsRef.current = clip.segments;
    segIndexRef.current = 0;
    setActiveKey(clipKey(clip));
    video.currentTime = clip.segments[0][0];
    void video.play();
  }, []);

  async function adjust(index: number, params: AdjustParams) {
    if (!projectId) return;
    const wasActive = detail?.clips[index] && clipKey(detail.clips[index]) === activeKey;
    try {
      const updated = await adjustClip(projectId, index, params);
      setDetail(updated);
      // If the edited clip was in the player, re-bind so playback uses the new bounds.
      const next = updated.clips[index];
      if (wasActive && next) preview(next);
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Could not adjust the clip");
    }
  }

  async function reprompt() {
    if (!projectId) return;
    setReprompting(true);
    try {
      const result = await repromptProject(projectId);
      navigate(`/processing/${result.job_id}`);
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Could not re-prompt");
      setReprompting(false);
    }
  }

  async function doExportProject() {
    if (!projectId) return;
    setExportingProject(true);
    try {
      const result = await exportProject(projectId);
      navigate(`/processing/${result.job_id}`);
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : "Could not export");
      message.error(detail);
      setExportingProject(false);
    }
  }

  async function doExport(index: number) {
    if (!projectId) return;
    try {
      const result = await exportClip(projectId, index);
      const withJson = result.transcript_filename ? " (+ transcription .json)" : "";
      message.success(`Exported “${result.filename}”${withJson} to your Downloads folder`);
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : "Export failed");
      message.error(detail);
    }
  }

  async function copyPrompt() {
    if (!projectId) return;
    try {
      const prompt = await getProjectPrompt(projectId);
      await navigator.clipboard.writeText(prompt);
      message.success("Prompt copied — paste it into your LLM, then paste its JSON back here");
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Could not copy the prompt");
    }
  }

  async function runImport() {
    if (!projectId) return;
    setImporting(true);
    try {
      const result = await importClips(projectId, importText);
      setDetail({ project: result.project, clips: result.clips });
      setImportOpen(false);
      setImportText("");
      message.success(`Imported ${result.imported} clip(s) — ${result.total} total`);
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : "Import failed");
      message.error(detail);
    } finally {
      setImporting(false);
    }
  }

  if (state === "loading") {
    return (
      <div className="surface flex min-h-[220px] items-center justify-center">
        <Spin />
      </div>
    );
  }
  if (state === "error" || !detail) {
    return <Alert type="error" showIcon message={error ?? "Project not found"} />;
  }

  const { project, clips } = detail;
  const activeClip = clips.find((c) => clipKey(c) === activeKey) ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <Link to="/" className="mb-1 inline-flex items-center gap-1 text-sm text-slate-500">
            <ArrowLeft aria-hidden size={15} /> Projects
          </Link>
          <h1 className="truncate text-2xl font-semibold tracking-normal text-slate-950">
            {project.title || project.id}
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Tag color={project.status === "done" ? "green" : "blue"} className="m-0">
            {project.status}
          </Tag>
          <Button
            icon={<FileVideo aria-hidden size={16} />}
            loading={exportingProject}
            disabled={!project.has_video || project.status !== "done"}
            onClick={() => void doExportProject()}
            title="Export the whole video as a DaVinci-ready .mov + transcription .json to Downloads"
          >
            Export video + transcript
          </Button>
          <Button
            icon={<ClipboardCopy aria-hidden size={16} />}
            disabled={project.status !== "done"}
            onClick={() => void copyPrompt()}
            title="Copy the prompt (instructions + transcript) to paste into a web LLM"
          >
            Copy prompt
          </Button>
          <Button
            icon={<FilePlus2 aria-hidden size={16} />}
            disabled={project.status !== "done"}
            onClick={() => setImportOpen(true)}
            title="Paste an LLM's JSON to add its clips to this list"
          >
            Import clips
          </Button>
          <Button
            icon={<Sparkles aria-hidden size={16} />}
            loading={reprompting}
            disabled={project.status !== "done"}
            onClick={() => void reprompt()}
            title="Re-run Gemini on the existing transcript"
          >
            Re-prompt Gemini
          </Button>
        </div>
      </div>

      <Modal
        title="Import clips from JSON"
        open={importOpen}
        onCancel={() => setImportOpen(false)}
        onOk={() => void runImport()}
        okText="Import"
        okButtonProps={{ loading: importing, disabled: !importText.trim() }}
        width={640}
      >
        <p className="mb-2 text-sm text-slate-600">
          Paste an LLM&apos;s reply in the judge JSON schema (line ranges + scores). Use{" "}
          <button
            type="button"
            className="font-medium text-blue-600 underline"
            onClick={() => void copyPrompt()}
          >
            Copy prompt
          </button>{" "}
          to feed the exact transcript to your LLM. Clips are grounded to timestamps and merged into
          this list (overlaps de-duped by score).
        </p>
        <Input.TextArea
          value={importText}
          onChange={(e) => setImportText(e.target.value)}
          placeholder={'{"clips": [{"start_line": 12, "end_line": 20, "hook": "…", "scores": {…}}]}'}
          autoSize={{ minRows: 8, maxRows: 18 }}
          spellCheck={false}
        />
      </Modal>

      {/* Sticky, centered player — stays in view while scrolling the list, so any clip plays
          in place without scrolling back up. */}
      {project.has_video ? (
        <div className="sticky top-2 z-20 mx-auto w-full max-w-2xl">
          <div className="surface overflow-hidden p-2 shadow-lg">
            <video
              ref={videoRef}
              controls
              preload="metadata"
              className="max-h-[45vh] w-full rounded-md bg-black"
              src={projectMediaUrl(project.id)}
            />
            <p className="mt-1 flex items-center gap-1.5 truncate px-1 text-sm text-slate-600">
              {activeClip ? (
                <>
                  {activeClip.clip_type === "compound" ? (
                    <Scissors aria-hidden size={14} className="shrink-0 text-purple-600" />
                  ) : null}
                  <span className="truncate">
                    Now playing: <span className="font-medium">{activeClip.hook}</span>
                  </span>
                </>
              ) : (
                <span className="text-slate-400">Press Preview on a clip to play it here</span>
              )}
            </p>
          </div>
        </div>
      ) : null}

      {project.status === "error" && project.error ? (
        <Alert type="error" showIcon message={project.error} />
      ) : null}

      {clips.length ? (
        <section className="grid gap-3">
          {clips.map((clip, i) => (
            <ClipCard
              key={clipKey(clip)}
              clip={clip}
              rank={i + 1}
              active={clipKey(clip) === activeKey}
              onPreview={project.has_video ? preview : undefined}
              onExport={project.has_video ? () => doExport(i) : undefined}
              onAdjust={
                project.status === "done" ? (params) => void adjust(i, params) : undefined
              }
            />
          ))}
        </section>
      ) : (
        <div className="surface flex min-h-[180px] items-center justify-center">
          <Empty
            description={
              project.status === "done" ? "No clips found" : "Still processing — no clips yet"
            }
          />
        </div>
      )}
    </div>
  );
}
