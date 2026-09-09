import { FormEvent, useRef, useState } from "react";
import { Button, Checkbox, Input, Tooltip, message } from "antd";
import { Link2, Send, Upload } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { createProject, createProjectFromFile } from "../services/api";
import { useAppStore } from "../store/useAppStore";

/**
 * The "new project" surface: paste a video URL and Start, or pick a local video file to upload.
 * Either path creates a project + queues its pipeline job, then navigates to the live progress.
 */
export function NewProjectBar() {
  const navigate = useNavigate();
  const upsertJob = useAppStore((state) => state.upsertJob);
  const fileInput = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [transcribeOnly, setTranscribeOnly] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = url.trim();
    if (!value) return;
    setSubmitting(true);
    try {
      const result = await createProject(value, transcribeOnly);
      upsertJob(result.job);
      setUrl("");
      navigate(`/processing/${result.job_id}`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Could not submit URL");
    } finally {
      setSubmitting(false);
    }
  }

  async function onFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = ""; // allow re-picking the same file later
    if (!file) return;
    setUploading(true);
    try {
      const result = await createProjectFromFile(file, transcribeOnly);
      upsertJob(result.job);
      navigate(`/processing/${result.job_id}`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Could not upload file");
    } finally {
      setUploading(false);
    }
  }

  return (
    <form className="surface flex flex-col gap-3 p-3" onSubmit={onSubmit}>
      <div className="flex flex-col gap-3 sm:flex-row">
        <Input
          size="large"
          aria-label="Video URL"
          prefix={<Link2 aria-hidden size={17} className="text-slate-400" />}
          placeholder="YouTube (or other) video URL"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
        />
        <Button
          size="large"
          type="primary"
          htmlType="submit"
          loading={submitting}
          icon={<Send aria-hidden size={17} />}
          className="shrink-0"
        >
          Start
        </Button>
        <Button
          size="large"
          type="default"
          loading={uploading}
          icon={<Upload aria-hidden size={17} />}
          className="shrink-0"
          onClick={() => fileInput.current?.click()}
        >
          Upload file
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(event) => void onFile(event)}
        />
      </div>
      <Tooltip title="Skip the Gemini clip finder — just download, transcribe, and save the word-level transcription JSON to your Downloads folder.">
        <Checkbox
          checked={transcribeOnly}
          onChange={(event) => setTranscribeOnly(event.target.checked)}
          className="w-fit text-sm text-slate-600"
        >
          Transcript only — skip the AI clip finder
        </Checkbox>
      </Tooltip>
    </form>
  );
}
