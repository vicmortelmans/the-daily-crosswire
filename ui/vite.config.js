import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://backend:8000",
      // Forward only requests for actual static files under /issues/
      // (e.g. /issues/{slug}/index.html, style.css, meta.json). Bare
      // SPA routes like /issues/:id and /issues/:id/details have no
      // extension and fall through to Vite's history fallback so
      // React Router can handle them on reload.
      // Optional query string allows the report iframe to cache-bust with ?v=.
      "^/issues/[^/]+/[^/]+\\.[a-z0-9]+(\\?.*)?$": "http://backend:8000",
    },
    host: "0.0.0.0",
    allowedHosts: true,
  },
});
