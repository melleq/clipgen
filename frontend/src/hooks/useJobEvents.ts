import { useEffect } from "react";

import { apiUrl } from "../services/api";
import type { ProgressEvent } from "../services/types";
import { useAppStore } from "../store/useAppStore";

function parseProgressEvent(event: MessageEvent<string>): ProgressEvent | null {
  try {
    return JSON.parse(event.data) as ProgressEvent;
  } catch {
    return null;
  }
}

export function useJobEvents(jobId: string | undefined) {
  const progress = useAppStore((state) => (jobId ? state.jobProgress[jobId] : undefined));
  const setConnection = useAppStore((state) => state.setJobConnection);
  const upsertEvent = useAppStore((state) => state.upsertJobEvent);

  useEffect(() => {
    if (!jobId) return;

    setConnection(jobId, "connecting");
    const source = new EventSource(apiUrl(`/api/jobs/${jobId}/events`));

    source.onopen = () => {
      setConnection(jobId, "open");
    };

    const handleMessage = (messageEvent: MessageEvent<string>) => {
      const event = parseProgressEvent(messageEvent);
      if (!event) return;
      upsertEvent(event);
      // "awaiting" pauses the run for chapter selection — like done/error, no more events follow.
      const status = event.detail.job_status;
      if (status === "done" || status === "error" || status === "awaiting") {
        source.close();
      }
    };

    source.addEventListener("progress", handleMessage);
    source.addEventListener("clips_ready", handleMessage);
    source.onerror = () => {
      setConnection(jobId, "error");
    };

    return () => {
      source.close();
    };
  }, [jobId, setConnection, upsertEvent]);

  return progress;
}
