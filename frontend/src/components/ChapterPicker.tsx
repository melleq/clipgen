import { Button, Checkbox, message } from "antd";
import { CheckCheck, ListX, Send } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { judgeProject } from "../services/api";
import type { Chapter } from "../services/types";
import { formatSeconds } from "../utils/format";

/**
 * The pause between transcribe and judge: the source video's chapters, each a checkbox. The
 * operator selects all / clears all / picks a subset, then sends only those chapters to Gemini.
 * Selecting every chapter sends the whole transcript (equivalent to not filtering at all).
 */
export function ChapterPicker({
  projectId,
  chapters,
  durationS,
}: {
  projectId: string;
  chapters: Chapter[];
  durationS: number;
}) {
  const navigate = useNavigate();
  // Chapters arrive start-sorted from the backend; the array index is the stable selection key the
  // backend expects (0-based, into the start-sorted list).
  const ordered = useMemo(
    () => [...chapters].sort((a, b) => a.start_s - b.start_s),
    [chapters],
  );
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(ordered.map((_, i) => i)),
  );
  const [sending, setSending] = useState(false);

  const allSelected = selected.size === ordered.length;
  const noneSelected = selected.size === 0;

  function toggle(i: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }

  // Each chapter runs to the next one's start (or the video's end). Blank when the end is unknown
  // (duration missing) or not after the start, so we never render a bogus "50:00 – 0:00".
  function rangeLabel(i: number): string {
    const start = formatSeconds(ordered[i].start_s);
    const next = ordered[i + 1];
    const end = next ? next.start_s : durationS;
    return end > ordered[i].start_s ? `${start} – ${formatSeconds(end)}` : start;
  }

  async function send() {
    setSending(true);
    try {
      // All selected → send the whole transcript (null); otherwise the chosen subset of indices.
      const selection = allSelected
        ? null
        : ordered.map((_, i) => i).filter((i) => selected.has(i));
      const result = await judgeProject(projectId, selection);
      navigate(`/processing/${result.job_id}`);
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Could not start the judge");
      setSending(false);
    }
  }

  return (
    <section className="mt-4 rounded-lg border border-slate-200 p-4">
      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Select chapters to send</h2>
          <p className="text-sm text-slate-600">
            Transcription is done. Pick the chapters to send to Gemini — deselect intros, ads, or
            anything you don&apos;t want clipped.
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button
            size="small"
            icon={<CheckCheck aria-hidden size={15} />}
            disabled={allSelected}
            onClick={() => setSelected(new Set(ordered.map((_, i) => i)))}
          >
            Select all
          </Button>
          <Button
            size="small"
            icon={<ListX aria-hidden size={15} />}
            disabled={noneSelected}
            onClick={() => setSelected(new Set())}
          >
            Clear all
          </Button>
        </div>
      </div>

      <ul className="grid gap-1.5">
        {ordered.map((ch, i) => (
          <li key={`${ch.start_s}-${i}`}>
            <Checkbox
              className="flex w-full items-center rounded-md border border-slate-200 px-3 py-2 hover:bg-slate-50"
              checked={selected.has(i)}
              onChange={() => toggle(i)}
            >
              <span className="font-mono text-xs text-slate-500">{rangeLabel(i)}</span>
              <span className="ml-3 text-sm text-slate-900">{ch.title || "(untitled)"}</span>
            </Checkbox>
          </li>
        ))}
      </ul>

      <div className="mt-4 flex items-center justify-between gap-3">
        <span className="text-sm text-slate-500">
          {selected.size} of {ordered.length} chapters selected
        </span>
        <Button
          type="primary"
          icon={<Send aria-hidden size={16} />}
          loading={sending}
          disabled={noneSelected}
          onClick={() => void send()}
        >
          Send to Gemini
        </Button>
      </div>
    </section>
  );
}
