import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider } from "antd";
import "antd/dist/reset.css";

import { AppRouter } from "./router";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        token: {
          borderRadius: 8,
          colorPrimary: "#2563eb",
          fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
        },
      }}
    >
      <AppRouter />
    </ConfigProvider>
  </React.StrictMode>,
);
