import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output lands in dist/, which services/configurator/app.py serves
// as static files (see the `_DIST` mount at the bottom of app.py) when it
// exists — this dev server on :5173 is for hot-reload iteration only.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
