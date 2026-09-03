# SatQuery AI — Earth Intelligence Dashboard 🛰️✨

Welcome! This is the **SatQuery AI geospatial intelligence dashboard** — a liquid-glassmorphic, multitemporal satellite analysis interface for change detection, SAR radar analysis, and spatial reasoning.

It's the analytical workhorse that pairs with our 3D landing experience. Built for a Smart India Hackathon project and refined with a lot of care.

---

## 🌟 Key features

### 1. Liquid glassmorphism UI
- Specular inner reflections, deep frosted refraction, and adaptive opposite background lighting (deep cosmic dark mode for light glass cards, cool luminous daylight for dark glass cards).
- Zero-leak fixed sidebar with quick access to Dashboard, Map Overview, Analyses, Datasets, Projects, Reports, Alerts, and Shared workspaces.
- Seamless dual-icon glass theme toggle (`☀️ / 🌙`).

### 2. Multimodal spatial intake drawer
- **Single optical image mode** — upload single high-resolution optical rasters (`.jpg`, `.jpeg`, `.png`, `.tif`, `.geotiff`).
- **Multi-image pair mode** — dedicated dual upload slots for **T1 Baseline** (e.g. May 2023) and **T2 Target** (e.g. May 2025) rasters for bi-temporal ChangeFormer analysis.
- **Sentinel-1 SAR radar mode** — dedicated polarimetric slots for **VV Co-Pol** and **VH Cross-Pol** for all-weather flood and canopy volume assessment.

### 3. Interactive Leaflet GIS viewport
- Locked high-resolution satellite imagery (Esri World Imagery) centered on **Vignan University (`17.3425° N, 78.7168° E`)**.
- Accurately grounded change polygons for **Vegetation Gain (+18.7%)**, **Built-up Expansion (+12.3%)**, and **Water Retention (-3.6%)**.
- **Interactive mini-calendar** — monthly day grid with one-click orbit epoch switching between May 2023 baseline and May 2025 changes.
- **Magic wand tool** — click two points to draw custom AOI bounding boxes with a floating `[🪄 AOI Active | ✕]` removal badge.
- **Draggable split-swipe comparison** — centrally constrained bi-temporal swipe slider to inspect pre- vs. post-development changes side-by-side.

### 4. Multi-session history switcher
- Instant context switching between 3 example scenarios:
  1. **Vignan University Campus Expansion & Vegetation Survey** (ChangeFormer T4)
  2. **Hussain Sagar Algal Bloom & Water Quality Assessment** (Sentinel-2 Multi-band NDWI)
  3. **East Coast Mangrove Canopy Density Audit** (Sentinel-1 SAR Dual-Pol)
- Fully functional `+ New Chat` workflow with interactive prompt starter chips.

---

## 🚀 Quick start

```bash
git clone https://github.com/trushendarreddy-cell/SatQuery-AI-fronend-v4.git
cd SatQuery-AI-fronend-v4/llm/satquery-frontend-dashboard
python serve.py 3001
```
Open **[http://localhost:3001](http://localhost:3001)** in your browser.

> Python 3.8+ recommended, or use any static web server (`npx serve .`, VS Code Live Server, etc.).

---

## 📁 Repository structure

```text
llm/
├── frontend/                         # 3D landing page (PlayCanvas, EDOLUS-inspired)
│   ├── index.html
│   ├── serve.py
│   ├── files/assets/                 # .glb, .basis, .ogg, .mp4 assets
│   └── README.md
├── satquery-frontend-dashboard/      # Earth intelligence dashboard
│   ├── index.html                    # Self-contained single-page app
│   ├── serve.py                      # Lightweight Python dev server
│   ├── interface.md                  # 5-stage roadmap & implementation tracker
│   ├── INTEGRATION_SPEC_AND_ROADMAP.md  # LangGraph agent architecture & API specs
│   └── README.md
└── geochat_api.py                    # Local GeoChat LLM wrapper (see "Local LLM" below)
```

---

## 💡 Local LLM

The dashboard expects a vision-language backend on `http://localhost:8000` (or whatever host you configure). We wrote a small **`geochat_api.py`** wrapper that boots the **GeoChat** vision-language model on your local machine and exposes the HTTP routes the dashboard needs. Run it with:

```bash
python geochat_api.py
```

This is what powers the live captioning, VQA, and grounded reasoning demos. See the script for setup, environment variables, and model download instructions.

---

## 🙏 Acknowledgements & thanks

- **Dashboard glassmorphism and layout** by [Prakash (@PrakashMB-1213)](https://github.com/PrakashMB-1213) — original UI design and structure. Thanks, Prakash, for the solid foundation.
- **3D landing page** is a learning tribute to [EDOLUS](https://edolus.com/) — the cinematic 3D experience that inspired our PlayCanvas landing scene. Thank you, EDOLUS.
- **Local LLM** is based on [GeoChat](https://github.com/mbzuai-oryx/GeoChat) (MBZUAI Oryx). Thank you to the GeoChat authors for open-sourcing such a capable vision-language model for remote sensing.
- **Map tiles** — Esri World Imagery.
- Built for **Smart India Hackathon 2026**. ❤️

---

## 👤 Authors

- **Trushendar Reddy** — frontend integration, GeoChat wrapper, repo merge.
- **Prakash** — original dashboard layout, glassmorphism system, and integration spec.
