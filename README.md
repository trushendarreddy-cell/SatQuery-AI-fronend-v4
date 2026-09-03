# SatQuery AI — Frontend v4 by rushendar reddy

A geospatial intelligence frontend built for a Smart India Hackathon 2026 project, pairing a cinematic 3D landing experience with a serious earth-observation dashboard for multitemporal satellite change detection, SAR radar analysis, and spatial reasoning.

> "We stand on the shoulders of giants — the 3D landing was inspired by [EDOLUS](https://edolus.com/) and the local vision-language brain is built on the open-source [GeoChat](https://github.com/mbzuai-oryx/GeoChat) model from MBZUAI's Oryx lab. Huge thanks to both teams."


##  What's inside
In this v4 version, I fixed some bugs in the frontend dashboard and I ran this llm locally but yet to be integrated with the frontend,the llm is perfect for answer and image change detection and comparison and provide adequate results with evidence.Do check out **geochat_api** file.
```
llm/
├── frontend/                         # 3D WebGL2 landing page (PlayCanvas + GSAP)
│   ├── index.html
│   ├── files/assets/                 # .glb models, .basis textures, .ogg audio, .mp4 backgrounds
│   └── README.md
├── satquery-frontend-dashboard/      # Leaflet + glassmorphism earth-intelligence dashboard
│   ├── index.html
│   ├── serve.py
│   ├── interface.md                  # 5-stage roadmap & implementation tracker
│   ├── INTEGRATION_SPEC_AND_ROADMAP.md  # LangGraph agent architecture & API specs
│   └── README.md
└── geochat_api.py                    # Local GeoChat-7B vision-language API wrapper
```

##  Local LLM

`geochat_api.py` is a FastAPI microservice that loads the **GeoChat-7B** vision-language model on your local GPU and exposes three endpoints for the dashboard:

- `POST /chat` — single-image VQA / captioning.
- `POST /compare` — two-image bi-temporal comparison with pixel-diff and semantic reasoning.
- `POST /compare/visual` — returns a side-by-side PNG with red highlight overlay for the changed regions.

Run it with:

```bash
$env:GEOCHAT_API_KEY="your-secret-key"
python geochat_api.py
```

##  Quick start

```bash
# 3D landing
cd llm/frontend && python serve.py 3000

# Dashboard
cd llm/satquery-frontend-dashboard && python serve.py 3001

# Local GeoChat LLM
python geochat_api.py
```

## Acknowledgements

-  **3D landing reference:** [EDOLUS](https://edolus.com/) — studied, learned from, and built with gratitude.
-  **Local LLM:** [GeoChat](https://github.com/mbzuai-oryx/GeoChat) (MBZUAI Oryx) — open-source satellite VLM.
-  **Map tiles:** Esri World Imagery.
-  **Original dashboard layout:** [Prakash (@PrakashMB-1213)](https://github.com/PrakashMB-1213) and [shankar (@shankar791)](https://github.com/shankar791)

**By**
**T.Rushendar Reddy**
**AIML**
**Hyderabad**
