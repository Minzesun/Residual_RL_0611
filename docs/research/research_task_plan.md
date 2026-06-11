# Task Plan: Residual RL + PID Literature Landscape

## Goal
Turn the vague topic "residual reinforcement learning plus PID, learning an additional residual control input" into structured research questions, gap analysis, a preliminary research plan, and an executable literature landscape grounded first in the user's Zotero collection.

## Phases
- [x] Phase 1: Set up persistent research artifacts
- [x] Phase 2: Extract local project and Zotero evidence
- [x] Phase 3: Supplement with external academic search
- [x] Phase 4: Classify and synthesize literature
- [x] Phase 5: Draft structured questions, gaps, and research plan
- [x] Phase 6: Verify sources and deliver final paths

## Key Questions
1. What exact problem should the project study: residual action added to PID output, residual tuning of PID terms, or safety-constrained residual correction around a fixed PID baseline?
2. Which gaps are strongest for a graduate-level contribution in vibration control: safety constraints, sample efficiency, PID-residual interface design, transfer/generalization, or experimental evaluation?
3. Which papers in the user's Zotero collection are core, adjacent, or background?
4. What minimal experiment plan can test novelty without over-expanding the scope?

## Decisions Made
- Use Zotero/local evidence first, then external search for gaps and missing canonical papers.
- Keep the research framing focused on fixed PID baseline plus learned residual control signal, not RL-tuned PID gains, unless a paper is useful as contrast.
- Write reusable Markdown artifacts in the current project root.
- Treat RL-PID tuning literature as an important comparison group, not as the main method.
- Use vibration-table/seismic-wave tracking as the application-specific gap, because the residual RL collection is broad but the adjacent Zotero collections show separate vibration-table and PID evidence.

## Errors Encountered
- Zotero live database first returned `database is locked`; resolved by using SQLite `mode=ro&immutable=1` read-only access.
- First academic search call used `CrossRef`/`arXiv`; tool expects lowercase `crossref`/`arxiv`, so the query will be rerun.
- Academic MCP search then failed with `asyncio.run() cannot be called from a running event loop`; external supplementation was completed with web search plus local Zotero metadata/PDF extraction.

## Status
**Complete** - structured problem, gap analysis, research plan, Zotero evidence extraction, and literature matrix have been written to Markdown files in the project root.
