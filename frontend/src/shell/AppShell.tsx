import { Outlet, NavLink } from "react-router-dom";
import { Clapperboard } from "lucide-react";

/** Root application chrome: a slim header + a centered content container. */
export function AppShell() {
  return (
    <div className="app-shell">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 text-slate-950">
            <Clapperboard aria-hidden size={20} strokeWidth={2.2} />
            <span className="text-sm font-semibold tracking-normal">clipgen</span>
          </NavLink>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-5 sm:px-6">
        <Outlet />
      </main>
    </div>
  );
}
