from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import asyncio
import os
import base64
import hmac
import hashlib
import json
from datetime import datetime
from github import Github, GithubException
from openai import OpenAI
import logging
import re

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="AI Project Generator")

# Configuration
VALID_SECRET = os.getenv("VALID_SECRET", "default-secret")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AIPIPE_TOKEN")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")

# Validate required env vars on startup
if not VALID_SECRET or VALID_SECRET == "default-secret":
    logger.warning("VALID_SECRET not configured - using default (insecure)")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize clients
try:
    gh = Github(GITHUB_TOKEN)
    gh.get_user().login
    logger.info("✓ GitHub client initialized")
except Exception as e:
    logger.error(f"✗ GitHub connection failed: {e}")
    gh = None

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Pydantic models
class Attachment(BaseModel):
    name: str
    url: str
    content: Optional[str] = None

class Check(BaseModel):
    js: str  # JavaScript code to evaluate

class Round2Brief(BaseModel):
    brief: str
    checks: List[Check]
    attachments: Optional[List[Attachment]] = None

class TaskTemplate(BaseModel):
    id: str
    brief: str
    attachments: List[Attachment] = []
    checks: List[Check]
    round2: Optional[List[Round2Brief]] = None

class ApiRequest(BaseModel):
    secret: str
    email: str
    task: str
    brief: str
    nonce: str
    round: int = Field(default=1, ge=1, le=2)
    evaluation_url: str
    attachments: List[Attachment] = []
    checks: List[Check] = []  # JavaScript checks to verify
    task_template: Optional[Dict[str, Any]] = None  # Full task definition

# Helper functions
def verify_secret_timing_safe(submitted_secret: str) -> bool:
    """Verify secret with timing-safe comparison"""
    return hmac.compare_digest(submitted_secret, VALID_SECRET)

def sign_payload(payload: Dict[str, Any], secret: str) -> str:
    """Sign payload with HMAC-SHA256"""
    payload_str = json.dumps(payload, sort_keys=True)
    return hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

async def fetch_attachment_data(url: str, max_size: int = 10_000_000) -> str:
    """Fetch attachment content with validation"""
    try:
        if url.startswith("data:"):
            parts = url.split(",", 1)
            if len(parts) == 2:
                logger.info(f"Parsed data URL attachment")
                return parts[1]
            return ""
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            if len(response.content) > max_size:
                logger.error(f"Attachment too large: {len(response.content)} bytes")
                raise ValueError(f"Attachment exceeds {max_size} bytes")
            
            encoded = base64.b64encode(response.content).decode()
            logger.info(f"Fetched attachment from {url}: {len(response.content)} bytes")
            return encoded
            
    except Exception as e:
        logger.error(f"Error fetching attachment {url}: {e}")
        raise

def build_llm_prompt(task_brief: str, attachments: List[Dict], current_code: Optional[str] = None, round: int = 1, checks: List[str] = None) -> str:
    """Build detailed LLM prompt with task requirements and checks"""
    
    attachment_context = ""
    if attachments:
        attachment_context = "\n\n## ATTACHMENTS PROVIDED:\n"
        for att in attachments:
            content_preview = att.get('content', '')[:300]
            attachment_context += f"- **{att.get('name', 'unknown')}**: {content_preview}...\n"
    
    checks_context = ""
    if checks:
        checks_context = "\n\n## VALIDATION CHECKS (must pass):\n"
        for i, check in enumerate(checks, 1):
            # Replace template variables with generic placeholders for LLM
            check_sanitized = check.strip()
            check_sanitized = check_sanitized.replace('${seed}', 'SEED_VALUE')
            check_sanitized = check_sanitized.replace('${result}', 'EXPECTED_RESULT')
            checks_context += f"{i}. JavaScript: `{check_sanitized}`\n"
    
    modify_instruction = ""
    if round > 1 and current_code:
        modify_instruction = f"""
## ROUND {round} ENHANCEMENT

You are improving an existing application. Current code:
```html
{current_code[:2000]}...
```

Maintain all previous functionality while adding the new requirements."""
    
    prompt = f"""You are an expert web developer creating minimal, functional single-page applications.

## TASK REQUIREMENTS:
{task_brief}

{attachment_context}

{checks_context}

{modify_instruction}

## STRICT REQUIREMENTS:
1. Return ONLY valid, complete HTML5 code (no markdown, no explanations, no code fences)
2. Include inline CSS and JavaScript in a single file
3. Must be runnable immediately when opened
4. Use fetch() API to load and process attachment data
5. Responsive design (mobile-first, Bootstrap if mentioned)
6. Comprehensive error handling with user feedback
7. Semantic HTML5 with accessibility (ARIA labels)
8. All business logic works client-side
9. CRITICAL: Ensure ALL validation checks pass
10. Keep code MINIMAL but complete and functional

The generated code MUST satisfy ALL the validation checks provided.

Return ONLY the HTML, starting with <!DOCTYPE html> and ending with </html>
Do NOT include markdown, code fences, or explanations."""

    return prompt

async def generate_app_code_with_llm(
    brief: str,
    attachments: Optional[List[Dict[str, str]]] = None,
    current_code: Optional[str] = None,
    round: int = 1,
    checks: Optional[List[str]] = None
) -> str:
    """Generate minimal app code using LLM with task requirements"""
    
    try:
        logger.info(f"Generating code for round {round}...")
        
        prompt = build_llm_prompt(
            task_brief=brief,
            attachments=attachments or [],
            current_code=current_code,
            round=round,
            checks=checks or []
        )
        
        message = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.choices[0].message.content.strip()
        
        # Clean up response - remove markdown wrappers
        if response_text.startswith("```html"):
            response_text = response_text.split("```html")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Validate it's HTML
        if not response_text.startswith("<!DOCTYPE") and not response_text.startswith("<html"):
            logger.warning("Response doesn't start with HTML, adding DOCTYPE...")
            response_text = "<!DOCTYPE html>\n" + response_text
        
        logger.info(f"Generated {len(response_text)} characters of code")
        return response_text
        
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        fallback = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated App</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .error { color: #d32f2f; background: #ffebee; padding: 12px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="error">
        <h2>Generation Error</h2>
        <p>The application code could not be generated. Please try again.</p>
    </div>
</body>
</html>"""
        return fallback

async def verify_github_pages(pages_url: str, max_retries: int = 10) -> bool:
    """Verify GitHub Pages is deployed and returns 200 OK"""
    delay = 1
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(max_retries):
            try:
                logger.info(f"Checking GitHub Pages ({attempt + 1}/{max_retries}): {pages_url}")
                response = await client.get(pages_url)
                if response.status_code == 200:
                    logger.info(f"✓ GitHub Pages verified at {pages_url}")
                    return True
                else:
                    logger.warning(f"GitHub Pages returned {response.status_code}")
            except Exception as e:
                logger.warning(f"Pages check attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
    
    logger.error(f"GitHub Pages verification failed after {max_retries} attempts")
    return False

async def create_github_repo(email: str, task: str, brief: str, app_code: str, task_id: str = "") -> Dict[str, str]:
    """Create GitHub repo and push code with no auto-init and no SHA conflicts"""
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")

    try:
        user = gh.get_user()

        # Create unique repo name
        timestamp = datetime.now().strftime("%s")[-6:]
        task_slug = task.lower().replace('_', '-').replace(' ', '-')[:40]
        repo_name = f"{task_slug}-{timestamp}"

        logger.info(f"Creating GitHub repo: {repo_name}")


        repo = user.create_repo(
            name=repo_name,
            description=f"Task: {task_id or task} | {brief[:100]}",
            private=False,
            auto_init=False
        )
        logger.info(f"✓ Repository created: {repo.html_url}")

        await asyncio.sleep(1.5)

        branch = "main"


        logger.info("Adding index.html...")
        repo.create_file(
            path="index.html",
            message="Initial commit: Add index.html",
            content=app_code,
            branch=branch
        )
        logger.info("✓ index.html added")


        readme_content = f"""# {task}

## Task Summary
{brief}

**Task ID:** {task_id or task}  
**Author:** {email}  
**Generated:** {datetime.now().isoformat()}

## Live Demo
https://{user.login}.github.io/{repo_name}/
"""
        repo.create_file(
            path="README.md",
            message="Add README",
            content=readme_content,
            branch=branch
        )
        logger.info("✓ Added README.md")


        license_content = f"""MIT License

Copyright (c) {datetime.now().year} {email}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software")...
"""
        repo.create_file(
            path="LICENSE",
            message="Add MIT License",
            content=license_content,
            branch=branch
        )
        logger.info("✓ Added LICENSE")

  
        gitignore_content = """# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Secrets
.secrets/
config.json
secrets.json
credentials.json
"""
        repo.create_file(
            path=".gitignore",
            message="Add .gitignore",
            content=gitignore_content,
            branch=branch
        )
        logger.info("✓ Added .gitignore")

        # Enable GitHub Pages using API directly
        pages_url = f"https://{user.login}.github.io/{repo_name}/"
        try:
            # Use the REST API to enable Pages
            import requests
            
            pages_data = {
                "source": {
                    "branch": "main",
                    "path": "/"
                }
            }
            
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.post(
                f"https://api.github.com/repos/{user.login}/{repo_name}/pages",
                headers=headers,
                json=pages_data
            )
            
            if response.status_code in [201, 409]:  # 201=created, 409=already exists
                logger.info(f"✓ GitHub Pages enabled: {pages_url}")
            else:
                logger.warning(f"GitHub Pages response: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.warning(f"GitHub Pages setup warning: {e}")

        # Return repo information
        return {
            "repo_url": repo.html_url,
            "pages_url": pages_url,
            "repo_name": repo_name,
            "commit_sha": None
        }

    except GithubException as e:
        logger.error(f"GitHub error: {e.status} - {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub error: {str(e)}")
    except Exception as e:
        logger.error(f"Repo creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def update_github_repo(email: str, task: str, brief: str, checks: Optional[List[str]] = None) -> Dict[str, str]:
    """Update existing GitHub repo for round 2"""
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    try:
        user = gh.get_user()
        repos = user.get_repos()
        
        task_clean = task.lower().replace("_", "-").replace(" ", "-")
        target_repo = None
        
        # Find the most recent repo matching the task
        logger.info(f"Searching for repo matching task: {task_clean}")
        matching_repos = []
        
        for repo in repos:
            if task_clean in repo.name.lower():
                matching_repos.append(repo)
                logger.info(f"Found matching repo: {repo.name} (created: {repo.created_at})")
        
        if not matching_repos:
            raise HTTPException(status_code=404, detail=f"No repository found for task: {task}")
        
        # Get the most recent one
        target_repo = sorted(matching_repos, key=lambda r: r.created_at, reverse=True)[0]
        logger.info(f"Selected repo: {target_repo.name}")
        
        # Fetch current index.html
        logger.info("Fetching current index.html...")
        try:
            current_file = target_repo.get_contents("index.html", ref="main")
            current_code = current_file.decoded_content.decode()
            logger.info(f"✓ Fetched current code ({len(current_code)} chars)")
        except Exception as e:
            logger.error(f"Failed to fetch index.html: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch current code")
        
        # Generate updated code with Round 2 requirements
        logger.info("Generating Round 2 code...")
        updated_code = await generate_app_code_with_llm(
            brief=brief,
            current_code=current_code,
            round=2,
            checks=checks
        )
        logger.info(f"✓ Generated updated code ({len(updated_code)} chars)")
        
        # Update index.html
        logger.info("Updating index.html...")
        target_repo.update_file(
            path="index.html",
            message=f"Round 2: {brief[:60]}",
            content=updated_code,
            sha=current_file.sha,
            branch="main"
        )
        logger.info("✓ Updated index.html")
        
        # Update README.md
        logger.info("Updating README.md...")
        try:
            readme_file = target_repo.get_contents("README.md", ref="main")
            readme_content = readme_file.decoded_content.decode()
            
            # Append Round 2 info
            round2_section = f"""

## Round 2 Update ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

**Enhancement:** {brief}

**Updated by:** {email}
"""
            updated_readme = readme_content + round2_section
            
            target_repo.update_file(
                path="README.md",
                message="Round 2: Update README",
                content=updated_readme,
                sha=readme_file.sha,
                branch="main"
            )
            logger.info("✓ Updated README.md")
        except Exception as e:
            logger.warning(f"Failed to update README: {e}")
        
        # Get latest commit SHA
        commits = target_repo.get_commits()
        commit_sha = commits[0].sha
        
        pages_url = f"https://{user.login}.github.io/{target_repo.name}/"
        
        return {
            "repo_url": target_repo.html_url,
            "commit_sha": commit_sha,
            "pages_url": pages_url,
            "repo_name": target_repo.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update repo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def push_results_with_retry(
    evaluation_url: str,
    payload: Dict[str, Any],
    secret: str,
    max_retries: int = 8
) -> bool:
    """Push results with exponential backoff (1, 2, 4, 8, 16, 32, 64, 128 seconds)"""
    
    signature = sign_payload(payload, secret)
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": signature
    }
    
    delay = 1
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(max_retries):
            try:
                logger.info(f"Posting to evaluation URL (attempt {attempt + 1}/{max_retries})")
                response = await client.post(
                    evaluation_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info(f"✓ Successfully posted results")
                    return True
                else:
                    logger.warning(f"Evaluation URL returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
    
    logger.error(f"Failed after {max_retries} retries")
    return False

async def process_request(req: ApiRequest):
    """Process API request (Round 1 or Round 2)"""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"ROUND {req.round} - Task: {req.task}")
        logger.info(f"Email: {req.email}")
        logger.info(f"{'='*70}\n")
        
        # Process attachments
        processed_attachments = []
        if req.attachments:
            logger.info(f"Processing {len(req.attachments)} attachments...")
            for att in req.attachments:
                try:
                    content = await fetch_attachment_data(att.url)
                    processed_attachments.append({
                        "name": att.name,
                        "content": content[:500]
                    })
                    logger.info(f"✓ {att.name}")
                except Exception as e:
                    logger.error(f"✗ Failed to process {att.name}: {e}")
        
        # Extract check codes
        check_codes = [check.js for check in req.checks] if req.checks else []
        
        if req.round == 1:
            # Round 1: Generate new app
            logger.info("Generating application code...")
            app_code = await generate_app_code_with_llm(
                brief=req.brief,
                attachments=processed_attachments,
                round=1,
                checks=check_codes
            )
            
            logger.info("Creating GitHub repository...")
            repo_info = await create_github_repo(
                req.email,
                req.task,
                req.brief,
                app_code,
                task_id=req.task_template.get("id", "") if req.task_template else ""
            )
            
        else:
            # Round 2: Update existing app
            logger.info("Updating application...")
            repo_info = await update_github_repo(
                req.email,
                req.task,
                req.brief,
                checks=check_codes
            )
        
        # Verify GitHub Pages
        logger.info("Verifying GitHub Pages...")
        pages_ok = await verify_github_pages(repo_info["pages_url"])
        if not pages_ok:
            logger.warning(" GitHub Pages verification timed out")
        
        # Prepare evaluation payload
        payload = {
            "email": req.email,
            "task": req.task,
            "round": req.round,
            "nonce": req.nonce,
            "repo_url": repo_info["repo_url"],
            "commit_sha": repo_info["commit_sha"],
            "pages_url": repo_info["pages_url"]
        }
        
        logger.info("Posting results to evaluation URL...")
        success = await push_results_with_retry(
            req.evaluation_url,
            payload,
            VALID_SECRET
        )
        
        if success:
            logger.info(f"✓ Round {req.round} COMPLETED")
        else:
            logger.error(f"✗ Evaluation URL unreachable")
            
    except asyncio.TimeoutError:
        logger.error(f"✗ TIMEOUT - Processing exceeded 10 minutes")
    except Exception as e:
        logger.error(f"✗ ERROR: {e}", exc_info=True)

# Routes
@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "github": "✓" if gh else "✗",
            "llm": "✓" if OPENAI_API_KEY else "✗"
        }
    )

@app.post("/api-endpoint")
async def api_endpoint(req: ApiRequest, background_tasks: BackgroundTasks):
    """Main API endpoint - accepts task requests and processes asynchronously"""
    
    try:
        # Input validation
        if not req.brief or len(req.brief.strip()) == 0:
            raise HTTPException(status_code=400, detail="Brief cannot be empty")
        
        if len(req.brief) > 5000:
            raise HTTPException(status_code=400, detail="Brief too long (max 5000 chars)")
        
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Invalid email")
        
        if not req.task or len(req.task.strip()) == 0:
            raise HTTPException(status_code=400, detail="Task cannot be empty")
        
        # Verify secret (timing-safe)
        logger.info(f"Verifying secret for {req.email}")
        if not verify_secret_timing_safe(req.secret):
            logger.warning(f"✗ Invalid secret")
            raise HTTPException(status_code=401, detail="Invalid secret")
        
        logger.info(f"✓ Secret verified")
        
        # Queue background processing with timeout
        async def process_with_timeout():
            try:
                await asyncio.wait_for(process_request(req), timeout=600)
            except asyncio.TimeoutError:
                logger.error(f"Processing timed out after 10 minutes")
        
        background_tasks.add_task(process_with_timeout)
        
        # Return 200 immediately
        return JSONResponse(
            status_code=200,
            content={
                "status": "Processing",
                "message": f"Round {req.round} request processing",
                "task": req.task,
                "round": req.round,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "name": "AI Project Generator API",
            "version": "2.0.0",
            "description": "Processes task templates and generates applications",
            "status": "ready"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    logger.info(f"Starting server on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)
