# AgentGuard Dashboard

A small React (Vite + Tailwind) UI for viewing AgentGuard security events: a live feed, a
per-session timeline, and an alerts panel.

## Prerequisites

- Node.js 18+
- The AgentGuard audit API running (see the main README's "Running the API + Dashboard" section).
  By default the dev server proxies `/events`, `/sessions`, `/alerts`, and `/health` to
  `http://localhost:8000`.

## Run (development)

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:5173
```

## API authentication

If the audit API is started with `AGENTGUARD_API_KEY` set, the dashboard must send that key.
Create `dashboard/.env.local`:

```
VITE_AGENTGUARD_API_KEY=your-key-here
```

For local development you can instead run the API without `AGENTGUARD_API_KEY` (it runs in an
open dev mode and logs a warning). Never do that in production.

## Build

```bash
npm run build      # outputs to dist/
npm run preview    # serve the production build locally
```
