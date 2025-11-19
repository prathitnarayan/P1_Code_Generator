async def enable_github_pages(repo, user_login: str, repo_name: str) -> str:
    """Enable GitHub Pages using PyGithub with proper error handling"""
    pages_url = f"https://{user_login}.github.io/{repo_name}/"
    
    try:
        # Method 1: Use PyGithub's built-in Pages support
        try:
            # Check if Pages already exists
            pages = repo.get_pages_build()
            logger.info(f"✓ GitHub Pages already enabled: {pages_url}")
            return pages_url
        except GithubException as e:
            if e.status == 404:
                # Pages doesn't exist, create it
                logger.info("GitHub Pages not found, enabling...")
            else:
                raise
        
        # Method 2: Use requests with proper GitHub token
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
            "build_type": "legacy"  # Use legacy Jekyll build
        }
        
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
            error_msg = response.json().get('message', 'Unknown error')
            logger.error(f"✗ GitHub Pages validation error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Pages validation failed: {error_msg}")
        else:
            logger.error(f"✗ GitHub Pages API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to enable GitHub Pages: {response.status_code}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Failed to enable GitHub Pages: {e}")
        raise HTTPException(status_code=500, detail=f"Pages setup failed: {str(e)}")


async def verify_github_pages(pages_url: str, max_retries: int = 20, initial_delay: int = 10) -> bool:
    """
    Verify GitHub Pages is deployed with intelligent retry strategy
    
    GitHub Pages typically takes 1-5 minutes for first deployment
    """
    logger.info(f"Waiting {initial_delay}s for initial GitHub Pages build...")
    await asyncio.sleep(initial_delay)  # Initial wait for build to start
    
    delay = 5  # Start with 5 second delays
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                logger.info(f"Checking GitHub Pages ({attempt + 1}/{max_retries}): {pages_url}")
                response = await client.get(pages_url)
                
                if response.status_code == 200:
                    # Verify it's not a 404 page
                    content = response.text.lower()
                    if "404" not in content[:500] and "not found" not in content[:500]:
                        logger.info(f"✓ GitHub Pages verified and serving content!")
                        return True
                    else:
                        logger.warning("Page returned 200 but appears to be 404 page")
                        
                elif response.status_code == 404:
                    logger.info(f"Pages not ready yet (404), waiting...")
                else:
                    logger.warning(f"Unexpected status: {response.status_code}")
                    
            except httpx.TimeoutException:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Check failed: {str(e)[:100]}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay = min(delay + 2, 15)  # Gradually increase delay up to 15s
    
    logger.error(f"✗ GitHub Pages not accessible after {max_retries} attempts (~5 min)")
    return False


async def create_github_repo(email: str, task: str, brief: str, app_code: str, task_id: str = "") -> Dict[str, str]:
    """Create GitHub repo with improved Pages deployment"""
    if not gh:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    try:
        user = gh.get_user()
        timestamp = datetime.now().strftime("%s")[-6:]
        task_slug = re.sub(r'[^a-z0-9-]', '-', task.lower())[:40]
        repo_name = f"{task_slug}-{timestamp}"
        
        logger.info(f"Creating repository: {repo_name}")
        
        # Create repo WITHOUT auto_init to avoid conflicts
        repo = user.create_repo(
            name=repo_name,
            description=f"Task: {task_id or task} | {brief[:100]}",
            private=False,
            auto_init=False,  # Critical: prevents default branch issues
            has_issues=False,
            has_wiki=False,
            has_downloads=False
        )
        
        logger.info(f"✓ Repository created: {repo.html_url}")
        
        # Wait for repo to be fully ready
        await asyncio.sleep(2)
        
        # Create files
        branch = "main"
        
        logger.info("Creating index.html...")
        repo.create_file(
            path="index.html",
            message="Initial commit: Add generated application",
            content=app_code,
            branch=branch
        )
        logger.info("✓ index.html created")
        
        # Create README
        readme_content = f"""# {task}

## Task Description
{brief}

**Task ID:** {task_id or task}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Author:** {email}

## 🚀 Live Demo
[View Application](https://{user.login}.github.io/{repo_name}/)

## 📝 About
This project was automatically generated as part of an AI development challenge.

---
*Generated by AI Project Generator*
"""
        repo.create_file(
            path="README.md",
            message="Add README",
            content=readme_content,
            branch=branch
        )
        logger.info("✓ README.md created")
        
        # Small delay before enabling Pages
        await asyncio.sleep(1)
        
        # Enable GitHub Pages
        logger.info("Enabling GitHub Pages...")
        pages_url = await enable_github_pages(repo, user.login, repo_name)
        
        # Get commit SHA
        commits = list(repo.get_commits())
        commit_sha = commits[0].sha if commits else None
        
        return {
            "repo_url": repo.html_url,
            "pages_url": pages_url,
            "repo_name": repo_name,
            "commit_sha": commit_sha
        }
        
    except GithubException as e:
        logger.error(f"GitHub API error: {e.status} - {e.data}")
        raise HTTPException(status_code=500, detail=f"GitHub error: {e.data.get('message', str(e))}")
    except Exception as e:
        logger.error(f"Repository creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
