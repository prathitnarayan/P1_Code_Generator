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
import requests

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
    checks: List[Check] = []
    task_template: Optional[Dict[str, Any]] = None

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
        if not response_text.startswith("<!DOCTYPE html>") and not response_text.startswith("<html"):
            logger.warning("Response doesn't start with HTML tag, prepending...")
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
    <title>Generation Error</title>
</head>
<body>
    <h1>Generation Error</h1>
    <p>The application code could not be generated. Please try again.</p>
</body>
</html>"""
        return fallback

# NEW FUNCTIONS FOR SINGLE REPO DEPLOYMENT

async def ensure_llm_pages_repo(user) -> Any:
    """
    Ensure 'LLM-Pages' repo exists, create if it doesn't.
    Returns the repository object.
    """
    repo_name = "LLM-Pages"
    
    try:
        # Try to get existing repo
        repo = user.get_repo(repo_name)
        logger.info(f"✓ Found existing repo: {repo.html_url}")
        return repo
    except GithubException as e:
        if e.status == 404:
            # Repo doesn't exist, create it
            logger.info(f"Creating new repo: {repo_name}")
            repo = user.create_repo(
                name=repo_name,
                description="Automated LLM-generated web applications - code is replaced on each deployment",
                private=False,
                auto_init=False,  # Don't auto-init, we'll create files manually
                has_issues=False,
                has_wiki=False,
                has_downloads=False
            )
            logger.info(f"✓ Created repo: {repo.html_url}")
            
            # Wait for repo to be ready
            await asyncio.sleep(2)
            
            # Create initial README to establish main branch
            logger.info("Creating initial README...")
            repo.create_file(
                path="README.md",
                message="Initial commit",
                content="# LLM Pages\n\nAutomated deployment repository.",
                branch="main"
            )
            logger.info("✓ Initial README created")
            
            # Wait a bit for branch to be established
            await asyncio.sleep(2)
            
            return repo
        else:
            raise


async def enable_github_pages(repo, user_login: str, repo_name: str) -> str:
    """Enable GitHub Pages using REST API with proper error handling"""
    pages_url = f"https://{user_login}.github.io/{repo_name}/"
    
    # Enable Pages using REST API (skip PyGithub check - it's unreliable)
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    pages_data = {
        "source": {
            "branch": "main",
            "path": "/"
        },
        "build_type": "legacy"
    }
    
    try:
        logger.info(f"Enabling GitHub Pages via API...")
        response = requests.post(
            f"https://api.github.com/repos/{user_login}/{repo_name}/pages",
            headers=headers,
            json=pages_data,
            timeout=30
        )
        
        if response.status_code == 201:
            logger.info(f"✓ GitHub Pages enabled successfully: {pages_url}")
            return pages_url
        elif response.status_code == 409:
            logger.info(f"✓ GitHub Pages already exists: {pages_url}")
            return pages_url
        elif response.status_code == 422:
            error_data = response.json()
            error_msg = error_data.get('message', 'Unknown error')
            logger.error(f"✗ GitHub Pages API error 422: {error_msg}")
            logger.error(f"   Full response: {response.text}")
            raise HTTPException(status_code=500, detail=f"Pages API error: {error_msg}")
        else:
            logger.error(f"✗ GitHub Pages API error: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to enable GitHub Pages: HTTP {response.status_code}"
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Network error enabling Pages: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error enabling Pages: {e}")
        raise HTTPException(status_code=500, detail=f"Pages setup failed: {str(e)}")


async def check_pages_status(user_login: str, repo_name: str) -> dict:
    """Check if GitHub Pages is enabled and get its status"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    try:
        response = requests.get(
            f"https://api.github.com/repos/{user_login}/{repo_name}/pages",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Pages status: {data.get('status', 'unknown')}")
            return data
        elif response.status_code == 404:
            logger.warning("✗ Pages not configured")
            return None
        else:
            logger.warning(f"Pages check returned: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Could not check Pages status: {e}")
        return None


async def update_or_create_file(repo, path: str, content: str, message: str, branch: str = "main"):
    """
    Update file if it exists, create if it doesn't.
    Handles SHA conflicts automatically.
    """
    try:
        # Try to get existing file
        existing_file = repo.get_contents(path, ref=branch)
        
        # Update existing file
        repo.update_file(
            path=path,
            message=message,
            content=content,
            sha=existing_file.sha,
            branch=branch
        )
        logger.info(f"✓ Updated {path}")
        
    except GithubException as e:
        if e.status == 404:
            # File doesn't exist, create it
            repo.create_file(
                path=path,
                message=message,
                content=content,
                branch=branch
            )
            logger.info(f"✓ Created {path}")
        else:
            raise


async def deploy_to_llm_pages(email: str, task: str, brief: str, app_code: str, 
                               task_id: str = "", round: int = 1) -> Dict[str, str]:
    """
    Deploy application to the single 'LLM-Pages' repository.
    Replaces all code on each deployment.
    """
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    try:
        user = gh.get_user()
        
        # Ensure LLM-Pages repo exists
        repo = await ensure_llm_pages_repo(user)
        repo_name = repo.name
        
        logger.info(f"Deploying to {repo.html_url}")
        
        branch = "main"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # 1. Update index.html (main application)
        logger.info("Updating index.html...")
        await update_or_create_file(
            repo=repo,
            path="index.html",
            content=app_code,
            message=f"Deploy: {task} (Round {round}) - {timestamp}",
            branch=branch
        )
        
        # 2. Update README.md with deployment history
        logger.info("Updating README.md...")
        readme_content = f"""# LLM Pages
This repository hosts automatically generated web applications. The codebase is **completely replaced** on each deployment.
## 🚀 Current Deployment
**Task:** {task}  
**Task ID:** {task_id or 'N/A'}  
**Round:** {round}  
**Deployed:** {timestamp}  
**Author:** {email}
### Description
{brief}
## 🌐 Live Application
**[View Current App](https://prathitnarayan.github.io/{repo_name}/)**
---
## 📝 About This Repository
- **Purpose:** Automated deployment of LLM-generated applications
- **Behavior:** Each deployment completely replaces the previous codebase
- **Updates:** Code is regenerated on demand based on task requirements
- **Technology:** Single-page HTML applications with inline CSS/JS
## 🔄 Deployment History
This README is updated with each deployment. Previous deployments are tracked in commit history.
### Latest Changes
- **Task:** {task}
- **Brief:** {brief[:200]}{'...' if len(brief) > 200 else ''}
- **Round:** {round}
- **Timestamp:** {timestamp}
---
*Powered by AI Project Generator | Last updated: {timestamp}*
"""
        
        await update_or_create_file(
            repo=repo,
            path="README.md",
            content=readme_content,
            message=f"Update README: {task} - {timestamp}",
            branch=branch
        )
        
        # 3. Create/Update deployment metadata file (JSON)
        logger.info("Updating deployment metadata...")
        metadata = {
            "task": task,
            "task_id": task_id,
            "round": round,
            "brief": brief,
            "email": email,
            "deployed_at": timestamp,
            "deployment_count": "See commit history"
        }
        
        await update_or_create_file(
            repo=repo,
            path="deployment.json",
            content=json.dumps(metadata, indent=2),
            message=f"Update metadata: {task}",
            branch=branch
        )
        
        # 4. Ensure .gitignore exists
        gitignore_content = """# Environment
.env
.env.local
# IDE
.vscode/
.idea/
*.swp
# OS
.DS_Store
Thumbs.db
# Secrets
.secrets/
credentials.json
"""
        
        try:
            repo.get_contents(".gitignore", ref=branch)
            logger.info("✓ .gitignore exists")
        except GithubException as e:
            if e.status == 404:
                await update_or_create_file(
                    repo=repo,
                    path=".gitignore",
                    content=gitignore_content,
                    message="Add .gitignore",
                    branch=branch
                )
        
        # Get latest commit SHA
        commits = list(repo.get_commits(sha=branch))
        commit_sha = commits[0].sha if commits else None
        
        pages_url = f"https://prathitnarayan.github.io/{repo_name}/"
        
        # Ensure Pages is enabled (with retry)
        logger.info("Ensuring GitHub Pages is enabled...")
        max_enable_retries = 3
        pages_enabled = False
        
        for attempt in range(max_enable_retries):
            try:
                await enable_github_pages(repo, user.login, repo_name)
                pages_enabled = True
                break
            except Exception as e:
                logger.warning(f"Pages enable attempt {attempt + 1} failed: {e}")
                if attempt < max_enable_retries - 1:
                    await asyncio.sleep(2)
        
        if not pages_enabled:
            logger.error("✗ Failed to enable GitHub Pages after retries")
            logger.info("Please manually enable Pages at:")
            logger.info(f"  https://github.com/{user.login}/{repo_name}/settings/pages")
        
        logger.info(f"✓ Deployment complete")
        logger.info(f"  Repo: {repo.html_url}")
        logger.info(f"  Pages: {pages_url}")
        logger.info(f"  Commit: {commit_sha}")
        
        return {
            "repo_url": repo.html_url,
            "pages_url": pages_url,
            "repo_name": repo_name,
            "commit_sha": commit_sha
        }
        
    except GithubException as e:
        logger.error(f"GitHub API error: {e.status} - {e.data}")
        raise HTTPException(
            status_code=500, 
            detail=f"GitHub error: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def verify_github_pages(pages_url: str, max_retries: int = 20, initial_delay: int = 10) -> bool:
    """
    Verify GitHub Pages is deployed and accessible.
    For existing repos, Pages should be faster (30s-2min vs 1-5min for new repos)
    """
    logger.info(f"Waiting {initial_delay}s for GitHub Pages build...")
    await asyncio.sleep(initial_delay)
    
    delay = 5
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                logger.info(f"Verifying Pages ({attempt + 1}/{max_retries}): {pages_url}")
                response = await client.get(pages_url)
                
                if response.status_code == 200:
                    content = response.text.lower()
                    # Check it's not a GitHub 404 page
                    if "404" not in content[:500] and "not found" not in content[:500]:
                        logger.info(f"✓ GitHub Pages verified and serving content!")
                        return True
                    else:
                        logger.info("Received 200 but appears to be 404 page, waiting...")
                        
                elif response.status_code == 404:
                    logger.info(f"Pages not ready yet (404)")
                else:
                    logger.warning(f"Unexpected status: {response.status_code}")
                    
            except httpx.TimeoutException:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Verification attempt failed: {str(e)[:100]}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay = min(delay + 2, 15)
    
    logger.error(f"✗ Pages verification timed out after {max_retries} attempts")
    return False


async def push_results_with_retry(
    evaluation_url: str,
    payload: Dict[str, Any],
    secret: str,
    max_retries: int = 8
) -> bool:
    """Push results with exponential backoff"""
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
    """Process API request - now deploys to single LLM-Pages repo"""
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
        
        # For Round 2, fetch current code from LLM-Pages repo
        current_code = None
        if req.round == 2:
            try:
                logger.info("Fetching current code for Round 2 enhancement...")
                user = gh.get_user()
                repo = user.get_repo("LLM-Pages")
                current_file = repo.get_contents("index.html", ref="main")
                current_code = current_file.decoded_content.decode()
                logger.info(f"✓ Fetched current code ({len(current_code)} chars)")
            except Exception as e:
                logger.warning(f"Could not fetch current code: {e}")
        
        # Generate application code
        logger.info(f"Generating application code (Round {req.round})...")
        app_code = await generate_app_code_with_llm(
            brief=req.brief,
            attachments=processed_attachments,
            current_code=current_code,
            round=req.round,
            checks=check_codes
        )
        
        # Deploy to LLM-Pages repo (replaces everything)
        logger.info("Deploying to LLM-Pages repository...")
        repo_info = await deploy_to_llm_pages(
            email=req.email,
            task=req.task,
            brief=req.brief,
            app_code=app_code,
            task_id=req.task_template.get("id", "") if req.task_template else "",
            round=req.round
        )
        
        # Verify GitHub Pages
        logger.info("Verifying GitHub Pages deployment...")
        pages_ok = await verify_github_pages(repo_info["pages_url"])
        
        if not pages_ok:
            logger.warning("⚠ GitHub Pages verification timed out (may still deploy)")
        
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
        
        # Push results
        logger.info("Posting results to evaluation URL...")
        success = await push_results_with_retry(
            req.evaluation_url,
            payload,
            VALID_SECRET
        )
        
        if success:
            logger.info(f"✓ Round {req.round} COMPLETED SUCCESSFULLY")
        else:
            logger.error(f"✗ Failed to reach evaluation URL")
            
    except asyncio.TimeoutError:
        logger.error(f"✗ TIMEOUT - Processing exceeded time limit")
    except Exception as e:
        logger.error(f"✗ PROCESSING ERROR: {e}", exc_info=True)


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
            "status": "ready",
            "deployment": "Single LLM-Pages repository",
            "pages_url": "https://prathitnarayan.github.io/LLM-Pages/"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"GitHub Pages URL: https://prathitnarayan.github.io/LLM-Pages/")
    uvicorn.run(app, host="0.0.0.0", port=port)
