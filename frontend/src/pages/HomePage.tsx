import { Alert, Empty, Spin } from "antd";
import { useEffect } from "react";

import { NewProjectBar } from "../components/NewProjectBar";
import { ProjectCard } from "../components/ProjectCard";
import { useAppStore } from "../store/useAppStore";

export function HomePage() {
  const projects = useAppStore((state) => state.projects);
  const state = useAppStore((state) => state.projectsState);
  const error = useAppStore((state) => state.projectsError);
  const fetchProjects = useAppStore((state) => state.fetchProjects);

  useEffect(() => {
    void fetchProjects();
  }, [fetchProjects]);

  return (
    <div className="space-y-5">
      <section className="grid gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-slate-950">Projects</h1>
          <p className="mt-1 text-sm text-slate-600">
            Turn a video into a ranked list of clip-worthy moments.
          </p>
        </div>
        <NewProjectBar />
      </section>

      {error ? <Alert type="error" showIcon message={error} /> : null}

      {state === "loading" ? (
        <div className="surface flex min-h-[220px] items-center justify-center">
          <Spin />
        </div>
      ) : projects.length ? (
        <section className="clip-grid">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </section>
      ) : (
        <div className="surface flex min-h-[220px] items-center justify-center">
          <Empty description="No projects yet" />
        </div>
      )}
    </div>
  );
}
