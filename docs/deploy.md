# Deploying the demo

The target is a **Hugging Face Space** on the free CPU tier: one container
running the API and the Streamlit UI together, with the corpus downloaded and
indexed at build time so the first question is answered immediately.

Nothing here has been deployed. These are the steps to run with your own key,
and the files they refer to are in `deploy/huggingface/`.

---

## Why Hugging Face Spaces rather than Railway or Render

| | |
|---|---|
| **No card** | The free CPU tier needs no payment method. Railway's free allowance now expects one, and a demo that stops answering when a trial lapses is worse than no link in the README. |
| **One container, not two** | The app is an API plus a UI that talks to it over HTTP. On Spaces both run side by side and the API binds `127.0.0.1`, so the only public surface is the UI. The two-service hosts want two paid services, or one service and a public API nobody should be able to post PDFs to. |
| **Docker, not a buildpack** | The image bakes in the ~90MB embedding model and the built index. A buildpack host would fetch the model on first request, which looks like a hang to the one person the demo exists for. |
| **16GB RAM** | `torch` plus `chromadb` plus the model is comfortable. The 512MB tiers are not. |

The cost of the choice: a Space on the free tier **sleeps after 48 hours of
inactivity** and takes 30–60 seconds to wake. For a portfolio link that is
acceptable; for anything real it is not.

---

## Before you start

You need:

- A Hugging Face account — https://huggingface.co/join
- A Google Gemini API key — https://aistudio.google.com/apikey (free, no card)
- `git` and about 10 minutes, most of it the image build

Run `python scripts/check_models.py` locally first. It reports which models
your key can actually reach, and a key whose quota is already spent will
produce a Space that deploys perfectly and fails every question.

---

## Steps

### 1. Create the Space

Go to https://huggingface.co/new-space and set:

| Field | Value |
|---|---|
| Owner | your account |
| Space name | `policylens` |
| License | MIT |
| **Space SDK** | **Docker** → **Blank** |
| Hardware | CPU basic (free) |
| Visibility | Public |

Create it. You get an empty git repository at
`https://huggingface.co/spaces/<you>/policylens`.

### 2. Clone the Space and copy this project into it

The Space repo needs this project's files, with **two differences**: its
`Dockerfile` must be the Space one, and its `README.md` must carry the Space's
YAML front matter.

```bash
git clone https://huggingface.co/spaces/<you>/policylens hf-policylens
cd hf-policylens

# Copy the application, but not the local junk.
#   --exclude .git        the Space has its own history
#   --exclude data        the index is built inside the image, and the PDFs
#                         are large and license-restricted
#   --exclude .venv .env  never push a key
rsync -a --exclude .git --exclude data --exclude .venv --exclude .env \
      --exclude __pycache__ --exclude .pytest_cache \
      ../policylens/ ./

# The Space's Dockerfile replaces the root one -- see the comment at the top
# of deploy/huggingface/Dockerfile for why there are two.
cp deploy/huggingface/Dockerfile ./Dockerfile

# The Space's README front matter is what sets sdk: docker and app_port: 7860.
cp deploy/huggingface/SPACE_README.md ./README.md
```

On Windows without `rsync`, copy the folder in Explorer and delete `data`,
`.venv`, `.env`, `__pycache__` and `.pytest_cache` from the copy.

### 3. Push

```bash
git add -A
git commit -m "PolicyLens"
git push
```

The build starts automatically. Watch it under the **Logs** tab. Expect
**8–15 minutes** the first time: `torch` is most of it, then the embedding
model, then the corpus download and the 279-chunk ingest.

### 4. Add the key

In the Space: **Settings → Variables and secrets → New secret**.

| Name | Value |
|---|---|
| `GOOGLE_API_KEY` | your Gemini key |

Add `GROQ_API_KEY` too if you have one — it is the last link in the fallback
chain.

**Secret, not variable.** A variable is visible to anyone who can see the
Space.

Then **Settings → Factory rebuild**, or restart the Space, so the container
picks the secret up.

### 5. Check it

The UI appears at `https://huggingface.co/spaces/<you>/policylens`. In order:

1. The sidebar should report the index as populated — 279 chunks, 3 documents.
   If it says the index is empty, a document host was down during the build;
   restart the Space and the entrypoint retries the download.
2. Ask **"how much is the minimum monthly wage in rupees?"** Expect
   `INSUFFICIENT_EVIDENCE`. This is the thing worth demonstrating.
3. Ask **"what is the maximum probation period for a worker?"** Expect
   `ANSWERED` with one verified citation to Section 13.

### 6. Put the link in the README

Replace the demo placeholder at the top of `README.md` with the Space URL,
and drop a recording at `docs/demo.gif`.

---

## If it goes wrong

| Symptom | Cause |
|---|---|
| Build fails on `pip install` | Almost always memory or a transient index failure. Factory rebuild. |
| Space runs, every question errors | No key, or the day's quota is spent. `/health` reports `llm_key_present`; `scripts/check_models.py` locally tells you which it is. |
| Sidebar says the index is empty | A third-party document host was down during the build. The build is deliberately tolerant of that (`\|\| true`) so a host outage cannot turn into no app at all. Restart to retry. |
| "Port not detected" | `app_port: 7860` is missing from the README front matter, or the README front matter was not copied at all. |
| Build takes 20+ minutes | Normal on a cold cache. Only the first one. |

---

## What is deliberately not deployed

**The `/documents` upload endpoint is not reachable from the internet.** The
API binds `127.0.0.1` inside the container, so only the UI next to it can
reach the API at all. Anyone who could reach `POST /documents` could have a C
extension parse a PDF they chose on a machine running someone else's key. For
a public demo the whole API stays private and the UI is the only surface.

**No persistent storage.** The index lives in the image. A restart serves the
same corpus; an uploaded document would not survive one. Spaces offer
persistent disks on paid tiers.

**No CI deployment.** Pushing to `main` on GitHub does not touch the Space.
Two repositories, deliberately: a Space push is a deliberate act, and the
credentials for automating it would have to live somewhere.
