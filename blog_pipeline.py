"""
AI Research Blog GTM Pipeline
==============================
Keeps the original LangGraph research + writing backend 100% intact.
Adds:
  - Daily topic list (AI research updates, last 7 days)
  - Front-matter injection for Jekyll / GitHub Pages
  - Git commit + push to <user>/github.io or any gh-pages branch
  - APScheduler cron job (runs once per day at a configurable hour)
  - CLI: `python blog_pipeline.py run` or `python blog_pipeline.py schedule`
"""

from __future__ import annotations

import argparse
import logging
import operator
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, List, Literal, Optional
from zoneinfo import ZoneInfo





# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env / config
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "") #yaad keep it for future use if needed, but currently using Groq for cost reasons
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")          
GITHUB_REPO_URL  = os.getenv("GITHUB_REPO_URL", "")       
GITHUB_BRANCH    = os.getenv("GITHUB_BRANCH", "main")
POSTS_SUBDIR     = os.getenv("POSTS_SUBDIR", "_posts")    
SCHEDULE_HOUR    = int(os.getenv("SCHEDULE_HOUR", "6"))  
SCHEDULE_TZ      = os.getenv("SCHEDULE_TZ", "Asia/Kolkata")

# ---------------------------------------------------------------------------
#backend code
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict


# ------------------------------------------------------------------
# 1) structure
# ------------------------------------------------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5, description="How many results to fetch per query (3–8).")


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ------------------------------------------------------------------
# 2) State 
# ------------------------------------------------------------------
class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str


# ------------------------------------------------------------------
# 3) LLM  
# ------------------------------------------------------------------
#llm = ChatOpenAI(model="gpt-4.1-mini") payment wale time
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)  


# ------------------------------------------------------------------
# 4) Router  
# ------------------------------------------------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- For open_book weekly roundup, include queries that reflect the last 7 days constraint.
"""


def router_node(state: State) -> dict:
    topic = state["topic"]

    
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"As-of date: {state['as_of']}\n\n"
                    "Decide:\n"
                    "- mode: open_book / hybrid / closed_book\n"
                    "- needs_research: true/false\n"
                    "- 3 search queries if needed\n\n"
                    "Respond in plain text."
                )
            ),
        ]
    )

    text = response.content.lower()

    
    if "open_book" in text:
        mode = "open_book"
        recency_days = 7
    elif "hybrid" in text:
        mode = "hybrid"
        recency_days = 45
    else:
        mode = "closed_book"
        recency_days = 3650

    needs_research = "true" in text or mode != "closed_book"

    today = state["as_of"]

    queries = [
        f"AI research papers published last 7 days {today}",
        f"new AI model releases announcements last week {today}",
        f"latest AI industry news and breakthroughs {today}",
        f"AI startup funding and launches last 7 days {today}",
        f"major AI company updates OpenAI Google Meta Anthropic {today}",
    ]

    return {
        "needs_research": needs_research,
        "mode": mode,
        "queries": queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    if state.get("needs_research"):
        return "research"
    return "orchestrator"

# ------------------------------------------------------------------
# 5) Research (Tavily)  (try wo serpertool tool)
# ------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- Extract/normalize published_at as ISO (YYYY-MM-DD) if you can infer it from title/snippet.
  If you can't infer a date reliably, set published_at=null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""
def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])[:10]
    max_results = 8
    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

   
    trimmed_results = [
    {
        "title": r.get("title", ""),
        "url": r.get("url", ""),
        "snippet": r.get("snippet", "")[:200],
    }
    for r in raw_results[:8]
    ]
    response = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{trimmed_results}\n\n"
                    "Extract useful sources as bullet points with title and URL."
                )
            ),
        ]
    )

    text = response.content

    # 🔥 simple parsing → EvidenceItem
    evidence = []
    for line in text.split("\n"):
        if "http" in line:
            url = line[line.find("http"):].strip()
            evidence.append(
                EvidenceItem(
                    title=line[:80],
                    url=url,
                    snippet=line
                )
            )

    return {"evidence": evidence}



# ------------------------------------------------------------------
# 6) Orchestrator  (original, unchanged)
# ------------------------------------------------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Flexibility:
- Do NOT use a fixed taxonomy unless it naturally fits.
- You may tag tasks (tags field), but tags are flexible.

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book (weekly news roundup):
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient fresh sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    # 🔥 LLM generates plan text
    response = llm.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:10]}\n\n"
                    "Create a blog outline with 5-7 sections. Each section should be short and clear."
                )
            ),
        ]
    )

    # 🔥 parse LLM output
    text = response.content.replace("#", "").strip()

    sections = [
        s.strip("- ").strip()
        for s in text.split("\n")
        if len(s.strip()) > 5
    ]

# 🔥 REMOVE DUPLICATES
    unique_sections = []
    for s in sections:
        if s not in unique_sections:
            unique_sections.append(s)

    sections = unique_sections[:5]

    
    if not sections:
        sections = [
            "Overview of recent AI developments",
            "Key model releases",
            "Important research papers",
            "Industry trends",
            "Developer implications",
        ]

    tasks = []
    for i, sec in enumerate(sections, 1):
        tasks.append(
            Task(
                id=i,
                title=sec[:80],
                goal="Explain and analyze this section",
                bullets=[
                "Key developments and details",
                "Underlying technical ideas",
                "Implications for AI systems",
                "What developers should pay attention to"
                ],
                target_words=180,
            )
        )

    plan = Plan(
        blog_title=state["topic"],
        audience="Developers",
        tone="Analytical",
        blog_kind="news_roundup" if mode == "open_book" else "explainer",
        tasks=tasks,
    )

    return {"plan": plan}  

# ------------------------------------------------------------------
# 7) Fanout  
# ------------------------------------------------------------------
def sequential_fanout(state: State):
    tasks = state["plan"].tasks
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in tasks[:2]   
    ]


# ------------------------------------------------------------------
# 8) Worker 
# ------------------------------------------------------------------
WORKER_SYSTEM = """You are a senior AI engineer and technical writer.

Write ONE section of a HIGH-QUALITY technical blog for developers.

CRITICAL RULES:
- Do NOT write generic statements (e.g., "AI is growing", "technology is evolving")
- Every paragraph must contain a specific insight or observation
- Focus on WHAT changed, WHY it matters, and WHAT it implies
- Avoid repetition across sentences
- Be analytical, not descriptive

STRUCTURE:
1. Start with a strong insight (not a generic intro)
2. Explain the development clearly
3. Add technical implications
4. Add developer-focused impact

STYLE:
- Sharp, concise, intelligent
- No fluff or filler
- Use precise technical language
- Prefer concrete examples over vague statements
- Add at least one non-obvious insight per section (something a beginner would miss)

GROUNDING:
- If mode == open_book:
  - ONLY use provided Evidence URLs for claims
  - Attach sources as: ([Source](URL))
  - If not supported → say "Not found in provided sources"

CODE:
- If requires_code == true → include a small working snippet

OUTPUT RULES:
- Markdown only
- Start with: ## <Section Title>
- No blog title
- No extra commentary
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")
    as_of = payload.get("as_of")
    recency_days = payload.get("recency_days")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:5]
        )

    import time

    # 🔥 RETRY LOGIC INSIDE FUNCTION
    for attempt in range(3):
        try:
            section_md = llm.invoke(
                [
                    SystemMessage(content=WORKER_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Blog title: {plan.blog_title}\n"
                            f"Audience: {plan.audience}\n"
                            f"Tone: {plan.tone}\n"
                            f"Blog kind: {plan.blog_kind}\n"
                            f"Topic: {topic}\n"
                            f"Mode: {mode}\n"
                            f"As-of: {as_of}\n\n"
                            f"Section title: {task.title}\n"
                            f"Goal: {task.goal}\n"
                            f"Bullets:{bullets_text}\n\n"
                            f"Evidence:\n{evidence_text}\n"
                        )
                    ),
                ]
            ).content.strip()
            break

        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(15)
            else:
                raise e

    return {"sections": [(task.id, section_md)]}
# ------------------------------------------------------------------
# 9) Reducer  
# ------------------------------------------------------------------
def reducer_node(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("Reducer called without a plan.")

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"final": final_md}


# ------------------------------------------------------------------
# 10) Build graph 
# ------------------------------------------------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", sequential_fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# ------------------------------------------------------------------
# 11) Runner
# ------------------------------------------------------------------
def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "final": "",
        }
    )

    plan: Plan = out["plan"]
    log.info("=" * 80)
    log.info(f"TOPIC: {topic}")
    log.info(f"AS_OF: {out.get('as_of')}  RECENCY_DAYS: {out.get('recency_days')}")
    log.info(f"MODE: {out.get('mode')}")
    log.info(f"BLOG_KIND: {plan.blog_kind}")
    log.info(f"NEEDS_RESEARCH: {out.get('needs_research')}")
    log.info(f"QUERIES: {(out.get('queries') or [])[:6]}")
    log.info(f"EVIDENCE_COUNT: {len(out.get('evidence', []))}")
    log.info(f"TASKS: {len(plan.tasks)}")
    log.info(f"SAVED_MD_CHARS: {len(out.get('final', ''))}")
    log.info("=" * 80)

    return out


# ===========================================================================

# ===========================================================================

# ---------------------------------------------------------------------------
# A) Daily topic generator
#    Always an "open_book" AI weekly-roundup so the router fetches real news.
# ---------------------------------------------------------------------------
DAILY_TOPIC_TEMPLATE = (
    "New AI research updates and breakthroughs {as_of}: "
    "new model releases, key papers, industry news, and implications for developers"
)


def today_topic(as_of: Optional[str] = None) -> str:
    if as_of is None:
        as_of = date.today().isoformat()

    return (
        f"Latest AI research breakthroughs, model releases, and industry developments "
        f" {as_of}, with developer-focused insights"
    )


# ---------------------------------------------------------------------------
# B) Jekyll front-matter injector
# ---------------------------------------------------------------------------
def _slugify(title: str) -> str:
    """Turn a blog title into a URL-safe slug."""
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:80]  # keep it reasonable


def inject_front_matter(markdown: str, plan: Plan, as_of: str, tags: List[str]) -> str:
    """
    Prepend Jekyll YAML front-matter to the raw markdown body.

    The original reducer already returns:
        # Blog Title\n\n<body>

    We strip the H1 and use it as the `title:` in front-matter so Jekyll
    renders it properly.
    """
    lines = markdown.splitlines()
    title = plan.blog_title

    # Strip leading H1 if present (reducer always adds it)
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).lstrip("\n")
    else:
        body = markdown

    tag_list = ", ".join(f'"{t}"' for t in tags)
    front_matter = f"""---
layout: post
title: "{title.replace('"', "'")}"
date: {as_of}
categories: [AI, Research]
tags: [{tag_list}]
author: AI Research Bot
---

"""
    return front_matter + body



def save_post(final_md: str, plan: Plan, as_of: str, repo_dir: Path) -> Path:
    """Write the post into the Jekyll _posts directory."""
    posts_dir = repo_dir / POSTS_SUBDIR
    posts_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(plan.blog_title)
    filename = f"{as_of}-{slug}.md"
    filepath = posts_dir / filename

  
    all_tags: List[str] = []
    for task in plan.tasks:
        all_tags.extend(task.tags)
    unique_tags = list(dict.fromkeys(all_tags))  

    post_content = inject_front_matter(final_md, plan, as_of, unique_tags)
    filepath.write_text(post_content, encoding="utf-8")
    log.info(f"Post saved → {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# D) Git commit + push
# ---------------------------------------------------------------------------
def _run_git(args: List[str], cwd: Path) -> str:
    """Run a git command and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr}"
        )
    return result.stdout.strip()


def ensure_repo(repo_dir: Path) -> None:
    """Clone or pull the GitHub Pages repo into repo_dir."""
    if not GITHUB_REPO_URL:
        raise ValueError(
            "GITHUB_REPO_URL is not set. "
            
            "GITHUB_REPO_URL=https://github.com/youruser/youruser.github.io"
        )

    # Embed token into URL for auth (HTTPS + PAT)
    auth_url = GITHUB_REPO_URL
    if GITHUB_TOKEN and "github.com" in auth_url:
        auth_url = auth_url.replace(
            "https://github.com",
            f"https://{GITHUB_TOKEN}@github.com",
        )

    if (repo_dir / ".git").exists():
        log.info("Repo already cloned — pulling latest…")
        _run_git(["pull", "origin", GITHUB_BRANCH], cwd=repo_dir)
    else:
        log.info(f"Cloning {GITHUB_REPO_URL} → {repo_dir} …")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH, auth_url, str(repo_dir)],
            check=True,
        )


def commit_and_push(repo_dir: Path, post_path: Path, as_of: str) -> None:
    """Stage the new post file, commit, and push."""
    rel_path = post_path.relative_to(repo_dir)

    _run_git(["add", str(rel_path)], cwd=repo_dir)
    _run_git(
        ["commit", "-m", f"chore(blog): auto-publish AI research post {as_of}"],
        cwd=repo_dir,
    )

    # Push (embed token in remote URL so no interactive prompt)
    if GITHUB_TOKEN and GITHUB_REPO_URL and "github.com" in GITHUB_REPO_URL:
        auth_url = GITHUB_REPO_URL.replace(
            "https://github.com",
            f"https://{GITHUB_TOKEN}@github.com",
        )
        _run_git(["push", auth_url, GITHUB_BRANCH], cwd=repo_dir)
    else:
        _run_git(["push", "origin", GITHUB_BRANCH], cwd=repo_dir)

    log.info(f"✅  Post pushed to GitHub Pages ({GITHUB_BRANCH})")


# ---------------------------------------------------------------------------
# E) Full daily pipeline
# ---------------------------------------------------------------------------
REPO_DIR = Path("./gh_pages_repo")  # local clone of the GitHub Pages repo


def daily_pipeline(as_of: Optional[str] = None) -> None:
    """
    1. Ensure local clone of GitHub Pages repo is up-to-date.
    2. Generate today's topic (AI research roundup).
    3. Run the full LangGraph research + writing pipeline.
    4. Inject Jekyll front-matter.
    5. Save to _posts/.
    6. Commit + push.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    log.info(f"=== Daily AI Blog Pipeline — {as_of} ===")

    # Step 1 — sync repo
    ensure_repo(REPO_DIR)

    # Step 2 — build topic
    topic = today_topic(as_of)
    log.info(f"Topic: {topic}")

    # Step 3 — run pipeline (original backend, zero changes)
    out = run(topic=topic, as_of=as_of)
    plan: Plan = out["plan"]
    final_md: str = out["final"]

    # Step 4+5 — save post with front-matter
    post_path = save_post(final_md, plan, as_of, REPO_DIR)

    # Step 6 — commit + push
    commit_and_push(REPO_DIR, post_path, as_of)

    log.info(f"=== Pipeline complete for {as_of} ===")


# ---------------------------------------------------------------------------
# F) Scheduler (learn more about it)
# ---------------------------------------------------------------------------
def start_scheduler() -> None:
    """Block and run the daily pipeline every day at SCHEDULE_HOUR."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        log.error("APScheduler not installed. Run: uv add apscheduler")
        sys.exit(1)

    tz = ZoneInfo(SCHEDULE_TZ)
    scheduler = BlockingScheduler(timezone=tz)

    scheduler.add_job(
        daily_pipeline,
        trigger="cron",
        hour=SCHEDULE_HOUR,
        minute=0,
        id="daily_ai_blog",
        name="Daily AI Research Blog",
        replace_existing=True,
    )

    log.info(
        f"Scheduler started — will run daily at {SCHEDULE_HOUR:02d}:00 {SCHEDULE_TZ}. "
        "Press Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# G) CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Research Blog GTM — automated blog pipeline for GitHub Pages"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run once
    run_p = sub.add_parser("run", help="Run the pipeline once (today or a specific date)")
    run_p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="As-of date (default: today)",
    )

    # start scheduler
    sub.add_parser("schedule", help="Start the daily cron scheduler")

    # plain LangGraph run (original behaviour, no GitHub publishing)
    dev_p = sub.add_parser("dev", help="Dev mode: run pipeline + save .md locally, no git push")
    dev_p.add_argument("topic", nargs="?", default=None, help="Custom topic (default: AI roundup)")
    dev_p.add_argument("--date", default=None, metavar="YYYY-MM-DD")

    args = parser.parse_args()

    if args.cmd == "run":
        daily_pipeline(as_of=args.date)

    elif args.cmd == "schedule":
        start_scheduler()

    elif args.cmd == "dev":
        as_of = args.date or date.today().isoformat()
        topic = args.topic or today_topic(as_of)
        out = run(topic=topic, as_of=as_of)
        plan: Plan = out["plan"]
        filename = f"{as_of}-{_slugify(plan.blog_title)}.md"
        all_tags = [t for task in plan.tasks for t in task.tags]
        post_content = inject_front_matter(out["final"], plan, as_of, list(dict.fromkeys(all_tags)))
        Path(filename).write_text(post_content, encoding="utf-8")
        log.info(f"Dev mode: post saved locally → {filename}")


if __name__ == "__main__":
    main()