# CLAUDE.md

## Project Overview

Interactive web application for exploring X/Twitter social network following graphs. Users input a username and the app retrieves their following network (1st and 2nd degree), computes influence metrics, detects communities via AI, and renders an interactive 3D force-directed graph.

## Tech Stack

- **Framework**: Streamlit 1.28.0
- **Visualization**: Three.js + Force-Graph-3D (CDN)
- **Twitter API**: RapidAPI (twitter283.p.rapidapi.com)
- **AI**: Google Gemini 2.0 Flash (topic extraction, community detection)
- **Graph**: NetworkX 3.1 (PageRank/CloutRank, community detection)
- **Async**: aiohttp for parallel API calls

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

API keys are in `config.py`: `RAPIDAPI_KEY`, `GEMINI_API_KEY`

## Directory Structure

```
follow_goblin_4/
├── app.py                    # Main Streamlit app (1,100+ lines)
├── config.py                 # API credentials + algorithm params
├── requirements.txt
├── api/
│   ├── twitter_client.py     # Async Twitter API wrapper
│   └── ai_client.py          # Gemini API wrapper
├── data/
│   ├── network.py            # NetworkData graph structure
│   ├── analysis.py           # CloutRank, In-Degree computation
│   ├── processing.py         # Data collection orchestrator
│   └── communities.py        # AI community detection
├── visualization/
│   ├── network_3d.py         # 3D force-graph HTML/JS generation
│   └── tables.py             # Data table rendering
└── utils/
    ├── helpers.py            # Session state management
    └── logging_config.py
```

## Data Flow

1. **Network Collection** — async parallel fetch of following lists (1st + 2nd degree)
2. **Metric Computation** — In-Degree + CloutRank (PageRank with contribution tracking)
3. **Tweet Processing** (optional) — batch fetch + Gemini summarization
4. **Topic Extraction** (optional) — AI clusters tweets into topics
5. **Community Detection** (optional) — AI labels and classifies accounts into communities
6. **Visualization** — 3D force-directed graph + filterable tables + CSV export

## Key Modules

| Module | Purpose |
|--------|---------|
| `data/analysis.py` | `compute_cloutrank()` — PageRank with incoming/outgoing contribution tracking |
| `data/processing.py` | `collect_network_data()` — parallel 1st/2nd degree network fetching |
| `api/ai_client.py` | Gemini wrapper: summaries, topics, community labels, classification |
| `visualization/network_3d.py` | Generates HTML/JS for Force-Graph-3D with node types (cube/sphere/tetrahedron) |

## Key Parameters (config.py)

- CloutRank: damping=0.85, epsilon=1e-8, max_iter=100
- MAX_CONCURRENT_REQUESTS = 50
- DEFAULT_BATCH_SIZE = 20
- DEFAULT_FETCH_TIMEOUT = 300s

## Status

Complete and functional. Dual-licensed (non-commercial free, commercial requires agreement).
