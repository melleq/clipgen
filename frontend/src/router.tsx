import { Navigate, RouterProvider, createBrowserRouter } from "react-router-dom";

import { AppShell } from "./shell/AppShell";
import { HomePage } from "./pages/HomePage";
import { ProcessingPage } from "./pages/ProcessingPage";
import { ClipsPage } from "./pages/ClipsPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "processing/:jobId", element: <ProcessingPage /> },
      { path: "p/:projectId", element: <ClipsPage /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);

export function AppRouter() {
  return <RouterProvider router={router} />;
}
