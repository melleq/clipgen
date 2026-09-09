# clipgen

Turn a long-form video into a ranked list of clip-worthy moments — driven entirely from a small
web UI. Give it a **YouTube URL** or a **local video file**; it downloads, transcribes (WhisperX,
optional speaker diarization), sends the numbered transcript to **Gemini** to pick the moments,
and grounds the result into a list of clips (real timestamps + transcript + per-dimension scores)
you can browse and preview.

This is a focused fork of `clips_autom` — the detection front-half only. No reframing, no
rendering, no editor: the output is a **data list** (`clips.json`), not cut video files.

## Layout

- `src/clipgen/` — the pipeline: `ingest` (yt-dlp / local file) → `transcribe` (WhisperX) →
  `judge` (Gemini) → `clips.ground` → `clips.json`. `server/` is the FastAPI app + a
  single-worker background job runner with SSE progress.
- `frontend/` — Vite + React + antd web UI: a Projects grid, a "new project" form (URL or file
  upload), live pipeline progress, and a read-only clips view with a preview player.
- `models/` — copied model weights (git-ignored; see below).
- `data/` — one directory per project (git-ignored): `meta.json`, `video.mp4`, `audio.wav`,
  `transcript.json`, `clips.json`, plus the exact judge prompt/response for review.

## Setup

1. **Python deps** (uv):
   ```bash
   uv sync
   ```
2. **Models** — copy the ASR / diarization / alignment weights into `./models`:
   ```bash
   ./scripts/copy_models.sh
   ```
3. **Config** — copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The example already
   points `TRANSCRIBE_MODEL` / `HF_HOME` / `TORCH_HOME` at `./models`. Set `HF_TOKEN` too if you
   want speaker diarization (needs the gated pyannote model terms accepted).
4. **Frontend deps**:
   ```bash
   cd frontend && pnpm install
   ```

## Run

Two processes (the Vite dev server proxies `/api` to the backend):

```bash
# terminal 1 — API + pipeline worker
uv run uvicorn clipgen.server.app:app --reload --port 8000

# terminal 2 — web UI
cd frontend && pnpm dev
```

Open the printed Vite URL, paste a video URL (or upload a file), watch the three-stage progress,
and land on the clip list when it finishes. From there, **Export** turns any clip into a
DaVinci-ready `.mov` (DNxHR + PCM) plus a sidecar transcription `.json` in your Downloads folder.

Tick **Transcript only — skip the AI clip finder** before starting to run ingest → transcribe and
export the whole video as a DaVinci-ready `.mov` plus its word-level transcription `.json` to
Downloads, with no Gemini pass. You can re-run this later from a project via **Export video +
transcript**.

## Tests

```bash
uv run ruff check . && uv run pytest
```
