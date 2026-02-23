import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="GitHub Repo Summarizer")

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
MAX_TOTAL_CHARS = 60_000
MAX_FILE_CHARS = 5_000

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".pdf",
    ".zip", ".tar", ".gz", ".lock", ".pyc", ".pyo",
    ".class", ".o", ".so", ".dll", ".exe",
    ".mp4", ".mp3", ".wav",
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock",
    "pipfile.lock", "composer.lock", "cargo.lock", "gemfile.lock",
    ".gitignore", ".gitattributes", ".editorconfig",
    "thumbs.db", ".ds_store",
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    "dist", "build", ".venv", "venv", "env",
    "vendor", ".idea", ".vscode", "coverage",
}

PRIORITY_FILES = [
    "readme.md", "readme.rst", "readme.txt", "readme",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "cargo.toml", "go.mod",
    "requirements.txt", "pipfile", "gemfile",
    "dockerfile", "docker-compose.yml",
    "main.py", "app.py", "index.py", "server.py",
    "index.js", "index.ts", "app.js", "app.ts",
    "main.go", "main.rs",
]


class SummarizeRequest(BaseModel):
    github_url: str


def parse_github_url(url: str):
    match = re.search(r"github\.com/([^/]+)/([^/?\s#]+)", url)
    if not match:
        return None, None
    owner = match.group(1)
    repo = match.group(2).rstrip("/").removesuffix(".git")
    return owner, repo


def should_skip(path: str) -> bool:
    parts = path.lower().split("/")
    for part in parts[:-1]:
        if part in SKIP_DIRS:
            return True
    filename = parts[-1]
    if filename in SKIP_FILENAMES:
        return True
    for ext in SKIP_EXTENSIONS:
        if filename.endswith(ext):
            return True
    if filename.startswith("."):
        return True
    return False


def get_priority(path: str) -> int:
    filename = path.lower().split("/")[-1]
    depth = path.count("/")
    if filename in PRIORITY_FILES:
        return depth
    for ext in {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".php"}:
        if filename.endswith(ext):
            return 10 + depth
    for ext in {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}:
        if filename.endswith(ext):
            return 20 + depth
    return 30 + depth


async def fetch_repo_contents(owner: str, repo: str):
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers
        )
        if r.status_code == 404:
            raise ValueError(f"Repository not found: {owner}/{repo}")
        if r.status_code == 403:
            raise ValueError("Repository is private or rate limit exceeded")
        r.raise_for_status()
        repo_data = r.json()
        branch = repo_data.get("default_branch", "main")

        tree_r = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
            headers=headers
        )
        tree_r.raise_for_status()
        tree = tree_r.json()

        all_files = [
            i["path"] for i in tree.get("tree", []) if i["type"] == "blob"
        ]
        all_dirs = sorted([
            i["path"] for i in tree.get("tree", []) if i["type"] == "tree"
        ])

        filtered = sorted(
            [f for f in all_files if not should_skip(f)],
            key=get_priority
        )

        files_content = {}
        total_chars = 0

        for path in filtered:
            if total_chars >= MAX_TOTAL_CHARS:
                break
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            resp = await client.get(raw_url)
            if resp.status_code != 200:
                continue
            try:
                content = resp.text
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n... [truncated]"
                files_content[path] = content
                total_chars += len(content)
            except Exception:
                continue

        return files_content, all_dirs, repo_data


def build_prompt(repo_data: dict, files: dict, dirs: list) -> str:
    name = repo_data.get("name", "")
    desc = repo_data.get("description", "") or ""
    top_dirs = [d for d in dirs if "/" not in d][:20]

    parts = [
        "Analyze this GitHub repository and return a structured JSON summary.",
        f"\nRepository name: {name}",
    ]
    if desc:
        parts.append(f"Description: {desc}")
    if top_dirs:
        tree_str = "\n".join(f"  {d}/" for d in top_dirs)
        parts.append(f"\nTop-level structure:\n{tree_str}")

    parts.append(f"\n--- File contents ({len(files)} files) ---")
    for path, content in files.items():
        parts.append(f"\n### {path}\n{content}")

    parts.append("""
Return ONLY valid JSON with exactly these fields:
{
  "summary": "2-4 sentences describing what the project does and its purpose",
  "technologies": ["list", "of", "languages", "frameworks", "libraries"],
  "structure": "1-3 sentences describing how the project is organized"
}
No markdown, no extra text. Only JSON.""")

    return "\n".join(parts)


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    owner, repo = parse_github_url(request.github_url)
    if not owner:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Invalid GitHub URL"})

    try:
        files, dirs, repo_data = await fetch_repo_contents(owner, repo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"status": "error", "message": str(e)})
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail={"status": "error", "message": f"GitHub API error: {e.response.status_code}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Failed to fetch repo: {e}"})

    if not files:
        raise HTTPException(status_code=422, detail={"status": "error", "message": "No readable files found"})

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail={"status": "error", "message": "NEBIUS_API_KEY not set"})

    llm = OpenAI(api_key=api_key, base_url=NEBIUS_BASE_URL)
    prompt = build_prompt(repo_data, files, dirs)

    try:
        response = llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a senior software engineer. Analyze repositories and return structured JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        return {
            "summary": result.get("summary", ""),
            "technologies": result.get("technologies", []),
            "structure": result.get("structure", ""),
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail={"status": "error", "message": "LLM returned invalid JSON"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"LLM error: {e}"})