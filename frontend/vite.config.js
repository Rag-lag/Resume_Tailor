import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api goes to the FastAPI server (uvicorn on :8000). On Vercel, the
// top-level rewrites in vercel.json route /api to the backend service instead.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
