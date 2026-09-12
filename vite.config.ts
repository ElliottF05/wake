import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

export default defineConfig({
  root: "playground",
  plugins: [basicSsl()],
  server: {
    https: true,
    host: true,
    headers: {
      // Cross-origin isolation required for full-precision timing results
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
