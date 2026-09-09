import { Button, Popconfirm, Tag, message } from "antd";
import { Clock, Film, ListChecks, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import type { ProjectStatus, ProjectSummary } from "../services/types";
import { useAppStore } from "../store/useAppStore";
import { formatSeconds } from "../utils/format";

const STATUS_COLOR: Record<ProjectStatus, string> = {
  queued: "default",
  ingesting: "blue",
  transcribing: "blue",
  awaiting: "gold",
  judging: "blue",
  done: "green",
  error: "red",
};

export function ProjectCard({ project }: { project: ProjectSummary }) {
  const navigate = useNavigate();
  const removeProject = useAppStore((state) => state.removeProject);
  const clipCount = project.n_clips ?? 0;
  const [deleting, setDeleting] = useState(false);

  async function remove() {
    setDeleting(true);
    try {
      await removeProject(project.id);
      message.success("Project deleted");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Could not delete project");
      setDeleting(false);
    }
  }

  const isDone = project.status === "done";

  return (
    <article className="surface flex min-h-[178px] flex-col justify-between p-4">
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <h2 className="line-clamp-2 text-base font-semibold leading-6 text-slate-950">
            {project.title || project.id}
          </h2>
          <Tag color={STATUS_COLOR[project.status] ?? "default"} className="m-0 shrink-0">
            {project.status}
          </Tag>
        </div>
        <div className="grid gap-2 text-sm text-slate-600">
          <div className="flex items-center gap-2">
            <Clock aria-hidden size={16} className="text-slate-400" />
            <span>{formatSeconds(project.duration_s)}</span>
          </div>
          <div className="flex items-center gap-2">
            <Film aria-hidden size={16} className="text-slate-400" />
            <span className="truncate">
              {clipCount > 0 ? `${clipCount} clips` : project.has_video ? "Media ready" : "No media"}
            </span>
          </div>
        </div>
        {project.status === "error" && project.error ? (
          <p className="line-clamp-2 text-sm text-red-600">{project.error}</p>
        ) : null}
      </div>
      <div className="mt-4 flex gap-2">
        <Button
          type={isDone ? "primary" : "default"}
          icon={<ListChecks aria-hidden size={17} />}
          onClick={() => navigate(`/p/${project.id}`)}
          className="flex-1"
        >
          View clips
        </Button>
        <Popconfirm
          title="Delete this project?"
          description="Removes its media, transcript and clips."
          okText="Delete"
          okButtonProps={{ danger: true }}
          onConfirm={() => void remove()}
        >
          <Button
            type="text"
            danger
            icon={<Trash2 aria-hidden size={17} />}
            loading={deleting}
            title="Delete project"
          />
        </Popconfirm>
      </div>
    </article>
  );
}
