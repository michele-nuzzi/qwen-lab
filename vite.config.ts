import { defineConfig } from 'vite';

export default defineConfig({
  // relative asset URLs, so the build works from any GitHub Pages path (/<repo>/)
  base: './',
  worker: { format: 'es' },
  // onnxruntime-web ships its own prebuilt bundle; esbuild pre-bundling it in dev breaks its wasm lookup
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1024 },
});
