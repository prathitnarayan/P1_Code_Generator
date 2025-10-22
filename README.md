=======
# AI Project Generator

A FastAPI-based AI application that generates single-page web applications from task briefs.

The app supports **Round 1** (initial app creation) and **Round 2** (enhancements/refactoring), automatically pushing code to GitHub and updating GitHub Pages.

## Live Deployment

The app is deployed on Hugging Face Spaces:

**https://huggingface.co/spaces/prathitnarayan/ai-project-generator**

You don't need to run it locally — just send API requests to the above URL.

## GitHub Repository

Generated apps are pushed to GitHub. Example repo for this project:

**https://github.com/prathitnarayan/P1_Code_Generator.git**

GitHub Pages are automatically enabled, so your generated app is live at:

**https://prathitnarayan.github.io/\<repo_name\>/**

## Workflow Diagram

```
          ┌──────────────┐
          │  Round 1     │
          │  Create App  │
          └─────┬────────┘
                │
                ▼
       Generate HTML + CSS + JS
                │
                ▼
       Push to GitHub Repository
                │
                ▼
      Enable GitHub Pages (auto)
                │
                ▼
      App live at pages_url
                │
                ▼
          ┌──────────────┐
          │  Round 2     │
          │ Enhance App  │
          └─────┬────────┘
                │
                ▼
       Add features / fix bugs
                │
                ▼
       Update GitHub repository
                │
                ▼
      GitHub Pages auto-updates
```

This illustrates the Round 1 → Round 2 → GitHub Pages flow.

##  API Endpoints

### Generate/Update App

```http
POST https://huggingface.co/spaces/prathitnarayan/ai-project-generator/api-endpoint
Content-Type: application/json
```

#### Request Body

```json
{
  "secret": "your-secret",
  "email": "you@example.com",
  "task": "weather-dashboard-app",
  "brief": "Add 7-day forecast and fix responsiveness issues",
  "nonce": "unique_nonce_002",
  "round": 2,
  "evaluation_url": "https://your-evaluation-endpoint.com",
  "checks": [
    { "js": "document.querySelector(\"#seven-day-forecast\")" }
  ],
  "attachments": [
    { "name": "forecast-data", "url": "https://example.com/forecast.json" }
  ]
}
```

**Parameters:**

- `round`: `1` for initial creation, `2` for enhancements
- `evaluation_url`: endpoint where GitHub URLs will be POSTed after generation
- `checks`: optional JavaScript snippets to validate app correctness
- `attachments`: optional external data or assets

#### Example curl Request

```bash
curl -X POST "https://huggingface.co/spaces/prathitnarayan/ai-project-generator/api-endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "secret": "GreenApple",
    "email": "25ds2000019@ds.study.iitm.ac.in",
    "task": "weather-dashboard-app",
    "brief": "Add 7-day forecast and fix responsiveness issues",
    "nonce": "unique_nonce_002",
    "round": 2,
    "evaluation_url": "https://your-evaluation-endpoint.com",
    "checks": [
      { "js": "document.querySelector(\"#seven-day-forecast\")" }
    ],
    "attachments": [
      { "name": "forecast-data", "url": "https://example.com/forecast.json" }
    ]
  }'

```

## How to Run Round 2 Workflow

### 1. Ensure Round 1 App Exists

You must have already created the initial app with `round: 1`.

### 2. Prepare Round 2 Request

- Update the `brief` with new features, bug fixes, or enhancements.
- Include `checks` to validate new functionality.
- Optionally include `attachments` (data files, images, JSON).

### 3. Submit the POST Request

See the curl example above.

### 4. Background Processing

- The server processes the request asynchronously.
- Generated code is pushed to the most recent GitHub repo for that task.

### 5. Check GitHub Pages

Your updated app is live at:

**https://prathitnarayan.github.io/\<repo_name\>/**

The server also POSTs metadata (repo URL, pages URL, commit SHA) to your `evaluation_url`.

## Notes

- All app code is generated in `index.html` with inline CSS/JS, ready to run.
- GitHub Pages is automatically enabled for each generated repo.
- Attachments are fetched via `fetch()` and included client-side.
- Secret is verified with timing-safe comparison for security.
- The API supports **Round 1** (create) and **Round 2** (enhance/update).






