import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App.tsx'
import './index.css'

const rootDocument = document.getElementById('root')

if (!rootDocument) {
  throw new Error('Root element not found')
}

createRoot(rootDocument).render(
  <StrictMode>
    <App />
  </StrictMode>
)
