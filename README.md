# Resume_Tailor
Resume tailoring application for individual Job description using Agentic AI.

A LangGraph pipeline (parse resume → analyze job description → retrieve evidence → tailor → score, with one retry) behind a FastAPI endpoint, with a React frontend. It only uses what's already on the resume and reports the gaps honestly.

## Run locally

```powershell
# Backend (needs OPENROUTER_API_KEY in .env)
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000

# Frontend, in a second terminal (proxies /api to :8000)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Deploy to Vercel

`vercel.json` deploys both parts as one project with Vercel Services: `/api/*` goes to the FastAPI backend, everything else to the React app.

1. Import the GitHub repo at vercel.com/new (or run `vercel` from the repo root).
2. Add `OPENROUTER_API_KEY` under Project Settings → Environment Variables.
3. Deploy.

Limits on the free Hobby plan: each run must finish within 300 seconds, so the pipeline skips its retry after 120 seconds. The embedding model (about 87 MB) is downloaded again on each cold start.
