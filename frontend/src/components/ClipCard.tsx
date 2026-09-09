import { Button, Popover, Tag, Tooltip, message } from "antd";
import { Copy, Download, Play, Scissors, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { AdjustParams } from "../services/api";
import type { ClipItem } from "../services/types";
import { SCORE_DIMENSIONS } from "../services/types";
import { formatScore, formatSeconds } from "../utils/format";

const EDGES: Array<"start" | "end"> = ["start", "end"];

function AdjustControls({ onAdjust }: { onAdjust: (p: AdjustParams) => void }) {
  return (
    <div className="grid w-60 gap-2">
      <div className="text-xs text-slate-500">
        Move a boundary by a word or a whole sentence.
      </div>
      {EDGES.map((edge) => (
        <div key={edge} className="flex items-center gap-1.5">
          <span className="w-10 text-xs font-medium capitalize text-slate-600">{edge}</span>
          <Tooltip title={`${edge}: one word shorter`}>
            <Button size="small" onClick={() => onAdjust({ edge, unit: "word", direction: "trim" })}>
              −w
            </Button>
          </Tooltip>
          <Tooltip title={`${edge}: one word longer`}>
            <Button
              size="small"
              onClick={() => onAdjust({ edge, unit: "word", direction: "expand" })}
            >
              +w
            </Button>
          </Tooltip>
          <Tooltip title={`${edge}: one sentence shorter`}>
            <Button
              size="small"
              onClick={() => onAdjust({ edge, unit: "sentence", direction: "trim" })}
            >
              −sentence
            </Button>
          </Tooltip>
          <Tooltip title={`${edge}: one sentence longer`}>
            <Button
              size="small"
              onClick={() => onAdjust({ edge, unit: "sentence", direction: "expand" })}
            >
              +sentence
            </Button>
          </Tooltip>
        </div>
      ))}
    </div>
  );
}

/** One grounded clip, read-only: timing, the judge's hook/reason, scores, and the transcript. */
export function ClipCard({
  clip,
  rank,
  active = false,
  onPreview,
  onExport,
  onAdjust,
}: {
  clip: ClipItem;
  rank: number;
  active?: boolean;
  onPreview?: (clip: ClipItem) => void;
  onExport?: () => Promise<void> | void;
  onAdjust?: (params: AdjustParams) => void;
}) {
  const isCompound = clip.clip_type === "compound";
  const [exporting, setExporting] = useState(false);

  async function handleExport() {
    if (!onExport) return;
    setExporting(true);
    try {
      await onExport();
    } finally {
      setExporting(false);
    }
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(clip.text);
      message.success("Transcript copied");
    } catch {
      message.error("Copy failed");
    }
  }

  // Compound clips get a purple left-edge stripe; the clip currently in the player gets a ring.
  const cls = ["surface p-4 transition-shadow"];
  if (isCompound) cls.push("border-l-4 border-l-purple-500");
  if (active) cls.push("ring-2 ring-blue-500 ring-offset-1");

  return (
    <article className={cls.join(" ")}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-slate-400">#{rank}</span>
            <h3 className="text-base font-semibold leading-6 text-slate-950">
              {clip.hook || "Untitled clip"}
            </h3>
            {isCompound ? (
              <Tag color="purple" icon={<Scissors aria-hidden size={12} />} className="m-0">
                compound · {clip.segments.length} parts
              </Tag>
            ) : null}
            {clip.source === "imported" ? (
              <Tag color="gold" className="m-0">
                imported
              </Tag>
            ) : null}
            {active ? (
              <Tag color="blue" className="m-0">
                playing
              </Tag>
            ) : null}
          </div>
          <p className="mt-1 text-sm text-slate-600">
            {formatSeconds(clip.start_s)}–{formatSeconds(clip.end_s)} ·{" "}
            {Math.round(clip.duration_s)}s spoken
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tag color="blue" className="m-0">
            {formatScore(clip.content_score)}
          </Tag>
          {onAdjust ? (
            <Popover
              trigger="click"
              placement="bottomRight"
              content={<AdjustControls onAdjust={onAdjust} />}
            >
              <Button
                size="small"
                icon={<SlidersHorizontal aria-hidden size={15} />}
                title="Expand or trim this clip"
              >
                Adjust
              </Button>
            </Popover>
          ) : null}
          {onPreview ? (
            <Button
              size="small"
              type={active ? "primary" : "default"}
              icon={<Play aria-hidden size={15} />}
              onClick={() => onPreview(clip)}
            >
              {active ? "Replay" : "Preview"}
            </Button>
          ) : null}
          {onExport ? (
            <Button
              size="small"
              icon={<Download aria-hidden size={15} />}
              loading={exporting}
              onClick={() => void handleExport()}
              title="Export a DaVinci-ready .mov (DNxHR) + transcription .json to your Downloads folder"
            >
              Export
            </Button>
          ) : null}
        </div>
      </div>

      {isCompound ? (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-purple-700">
          <Scissors aria-hidden size={13} className="shrink-0" />
          <span>
            {clip.segments.map((s) => `${formatSeconds(s[0])}–${formatSeconds(s[1])}`).join("  +  ")}
            {clip.splice_reason ? ` — ${clip.splice_reason}` : ""}
          </span>
        </p>
      ) : null}

      {clip.hook_sentence ? (
        <p className="mt-3 border-l-2 border-slate-200 pl-3 text-sm italic text-slate-700">
          “{clip.hook_sentence}”
        </p>
      ) : null}
      {clip.reason ? <p className="mt-2 text-sm text-slate-600">{clip.reason}</p> : null}
      {clip.payoff ? (
        <p className="mt-2 text-sm text-slate-600">
          <span className="font-medium text-slate-700">Payoff:</span> {clip.payoff}
        </p>
      ) : null}
      {clip.share_trigger ? (
        <p className="mt-1 text-sm text-slate-600">
          <span className="font-medium text-slate-700">Why it travels:</span> {clip.share_trigger}
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap gap-1.5">
        {SCORE_DIMENSIONS.map((dim) => (
          <Tooltip key={dim} title={dim}>
            <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {dim.slice(0, 4)} {Math.round(clip.scores[dim] ?? 0)}
            </span>
          </Tooltip>
        ))}
      </div>

      <details className="mt-3">
        <summary className="cursor-pointer text-sm font-medium text-slate-700">
          Transcript
        </summary>
        <div className="mt-2 flex items-start gap-2">
          <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{clip.text}</p>
          <Button
            type="text"
            size="small"
            icon={<Copy aria-hidden size={15} />}
            onClick={() => void copyText()}
            title="Copy transcript"
          />
        </div>
      </details>
    </article>
  );
}
