import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Windows'ta 8000 portu bazen "stale" bir process tarafindan tutulabiliyor.
// Env verilmezse varsayilan olarak 8010'a git.
const apiTarget = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8010';
const wsTarget =
  process.env.VITE_WS_TARGET ??
  apiTarget.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: wsTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
