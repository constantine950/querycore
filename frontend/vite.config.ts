import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/search": "http://backend:8000",
      "/autocomplete": "http://backend:8000",
      "/analytics": "http://backend:8000",
      "/index": "http://backend:8000",
      "/health": "http://backend:8000",
    },
  },
});
