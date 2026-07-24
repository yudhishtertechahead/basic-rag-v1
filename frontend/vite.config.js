import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // Output directly to FastAPI's static folder so `npm run build`
    // immediately makes the app available at http://localhost:8000/ui
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    // During development, proxy API calls to the FastAPI server
    // so you can run `npm run dev` without CORS issues
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
