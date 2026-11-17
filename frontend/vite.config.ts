import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // El frontend siempre llama a /api/... (mismo origen). En desarrollo y en `vite preview` se reenvía al backend
  // quitando el prefijo; en producción lo hace el rewrite de vercel.json (vite preview NO aplica vercel.json).
  // Se usa 127.0.0.1 (no "localhost") porque en Windows "localhost" puede resolver a ::1.
  const apiProxy = {
    '/api': {
      target: env.DEV_API_TARGET || 'http://127.0.0.1:8000',
      changeOrigin: true,
      rewrite: (path: string) => path.replace(/^\/api/, ''),
    },
  }

  return {
    plugins: [react()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
    build: {
      rolldownOptions: {
        output: {
          // Chunks con nombre: lo común a todas las rutas (React, router) se separa de Leaflet, que solo usan las pestañas.
          codeSplitting: {
            groups: [
              { name: 'react', test: /node_modules\/(react|react-dom|scheduler)\// },
              { name: 'router', test: /node_modules\/react-router\// },
              { name: 'leaflet', test: /node_modules\/(leaflet|@react-leaflet|react-leaflet)\// },
              { name: 'd3', test: /node_modules\/(d3-[a-z-]+|internmap)\// },
            ],
          },
        },
      },
    },
  }
})
