# JITK — Just-in-Time Knowledge (Backend + Frontend)

A lightweight **retrieval + provenance** demo:
- **Input:** case fields (Claim Type, State, Policy Type, etc.)
- **Output:** relevant clauses/snippets from policy PDFs **with verifiable citations** (page + highlighted paragraph preview)

Repo structure:
- `backend/` → FastAPI + Postgres(pgvector) + PDF rendering + ingestion
- `frontend/` → Next.js UI (Suggestions + Answer + Citations)

---

## 1) Prerequisites

### Required
- **Python 3.10+** (recommended: 3.10 or 3.11)
- **Node.js 18+** (recommended: 18 LTS or 20)
- **Docker Desktop** (for Postgres + pgvector)

### Optional but helpful
- Git
- VS Code

---

## 2) Quickstart (Local Demo)

### Step A — Start the database (pgvector)

Open a terminal:

```bash
cd backend
docker compose up -d
```

Confirm DB is running:
``` bash
docker ps
```

You should see a container like backend-db-1 running.

Step B — Enable pgvector extension (one-time)

Run:
``` bash
docker exec -it backend-db-1 psql -U jitk -d jitk -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Verify:
```bash
docker exec -it backend-db-1 psql -U jitk -d jitk -c "\dx"
```

You should see vector in the list of installed extensions.

Step C — Backend setup (FastAPI)

In a new terminal:
```bash
cd backend
python -m venv .venv
```

Activate:

Windows PowerShell
```bash
.\.venv\Scripts\Activate.ps1
```

macOS / Linux
``` bash
source .venv/bin/activate
```

Upgrade pip:
```bash
python -m pip install --upgrade pip
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run backend:
``` bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend should now be at:
```bash
Health: http://127.0.0.1:8000/

Docs: http://127.0.0.1:8000/docs
```
Step D — Ingest a PDF (IMPORTANT)

If you don’t ingest documents, /suggest and /answer will return empty results.

From another terminal (backend venv activated):
``` bash
cd backend
python -m app.ingest <pdf_path> <doc_name>
```

Example:
``` bash
python -m app.ingest ../docs/Homeowners_Policy.pdf Homeowners_Policy
```

What ingestion does:

parses PDF into chunks (paragraph/clause)

stores:

text, page_number, para_id

bbox (x1,y1,x2,y2) in PDF coords

embedding (pgvector)

doc metadata (name + version hash)

✅ After ingestion, your system can retrieve and cite with highlight previews.

Step E — Frontend setup (Next.js)

In a new terminal:
```bash
cd frontend
npm install
```

Create a .env.local:
```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

Run the UI:

npm run dev


Open:

http://localhost:3000

## Policy Document (Demo PDF)

This project requires at least one policy PDF to be ingested so that `/suggest` and `/answer` return results.

 Download the demo policy PDF from Google Drive:  
**Policy PDF (Google Drive):** <https://drive.google.com/file/d/1GofzVkCiBAHQTasicl7WntDpkPoApb_s/view?usp=sharing>

### How to use it (recommended steps)

1) **Create a folder to store PDFs**
```bash
mkdir -p docs
```

Download the PDF and place it inside docs/

Download from the Drive link above

Save as:

docs/Homeowners_Policy.pdf (recommended name)

Ingest the PDF into the vector DB
From backend/ (with .venv activated):

python -m app.ingest ../docs/Homeowners_Policy.pdf Homeowners_Policy


Verify ingestion worked
Now open the UI and try:

Claim Type: Flood

State: Florida

Policy Type: HO-3

Top K: 6

You should start seeing suggestions and cited answers.



3) Usage
Suggestions Mode

Enter case fields

Click Get Suggestions

UI displays top-K relevant chunks with:

snippet

doc/page/para id

preview highlight

preview page

Answer + Citations Mode

Enter case fields

Click Generate Answer

UI shows:

a readable answer block

cited bullets (each bullet = one source)

highlight/page previews

4) API Endpoints

Base: http://127.0.0.1:8000

POST /suggest

Returns ranked policy snippets.

Example:
``` bash
curl -X POST http://127.0.0.1:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "C-1001",
    "user_id": "agent_7",
    "fields": {
      "Claim Type": "Flood",
      "State": "Florida",
      "Policy Type": "HO-3"
    },
    "top_k": 6
  }'
```
POST /answer

Returns grounded answer + cited bullets.
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "C-1001",
    "user_id": "agent_7",
    "fields": {
      "Claim Type": "Flood",
      "State": "Florida",
      "Policy Type": "HO-3"
    },
    "top_k": 6,
    "max_bullets": 4
  }'
```
GET /source/page

Renders a full page image.

Example:
```bash
http://127.0.0.1:8000/source/page?doc_name=Homeowners_Policy&doc_version=<hash>&page=4
```
GET /source/highlight

Renders the page + highlights the exact chunk.
Optional crop=1 crops around the highlighted region.

Example:
```bash
http://127.0.0.1:8000/source/highlight?doc_name=Homeowners_Policy&doc_version=<hash>&page=4&para_id=p004-g001&crop=1
```
5) Troubleshooting
A) “type vector does not exist”

You forgot pgvector extension.

Fix:
```bash
docker exec -it backend-db-1 psql -U jitk -d jitk -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Restart backend after enabling extension.

B) No results in UI

You likely haven’t ingested documents.

Run:
```bash
python -m app.ingest <pdf_path> <doc_name>
```

Then refresh UI.

C) Uvicorn not found

You’re either:

not inside .venv

or requirements install failed

Activate venv and reinstall:
``` bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
D) Frontend can’t call backend (CORS or wrong API base)

Ensure frontend .env.local has:
```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

Restart Next.js after changing .env.local.

E) Docker service name confusion (no such service: backend)

Your compose file likely only defines db, so docker compose restart backend will fail.
Restart DB like:
```bash
docker compose restart db
```
6) Deployment Notes (Optional)

This demo uses:

local file paths for PDFs

local Postgres (pgvector)

For a public deployment, you typically:

store PDFs in object storage (S3 / Cloudflare R2 / GCS)

run Postgres with pgvector as a managed DB

deploy FastAPI on Render/Fly.io/Railway

deploy Next.js on Vercel/Netlify

7) Assumptions / Not Covered

Authentication / RBAC

Multi-tenant doc access control

Real LLM rewriting (this demo focuses on retrieval + provenance)

Auto-refresh ingestion pipeline