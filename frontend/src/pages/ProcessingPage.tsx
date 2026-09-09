import { Alert } from "antd";
import { useParams } from "react-router-dom";

import { PipelineProgress } from "../components/PipelineProgress";
import { useJobEvents } from "../hooks/useJobEvents";

export function ProcessingPage() {
  const { jobId } = useParams();
  const progress = useJobEvents(jobId);

  if (!jobId) {
    return <Alert type="error" showIcon message="Missing job id" />;
  }

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-normal text-slate-950">Pipeline run</h1>
        <p className="mt-1 text-sm text-slate-600">Job {jobId}</p>
      </div>
      <PipelineProgress progress={progress} />
    </div>
  );
}
