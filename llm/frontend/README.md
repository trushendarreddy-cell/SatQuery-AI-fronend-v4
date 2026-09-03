# SatQuery AI — Frontend 🛰️✨

Hi there! Thanks for stopping by our repo. **SatQuery AI** is a geospatial intelligence frontend we built for a Smart India Hackathon project. It pairs a slick 3D landing experience with a serious earth-observation dashboard for change detection, SAR radar analysis, and spatial reasoning.

We had a lot of fun building it, and we couldn't have done it without some amazing open work from the community — see the **Acknowledgements** section below.

---

## 🌟 What's inside

### 1. `llm/frontend/` — 3D Interactive Landing (PlayCanvas)
A cinematic WebGL2 landing page that boots straight into a 3D scene. Built on the PlayCanvas engine with GSAP animations, custom shaders, multi-track audio, and a bunch of `.glb` models, video backgrounds, and Basis-compressed textures.

### 2. `llm/satquery-frontend-dashboard/` — Earth Intelligence Dashboard
A liquid-glassmorphic geospatial dashboard featuring:
- **Interactive Leaflet map** with high-res Esri satellite imagery (centered on Vignan University, India).
- **Multi-modal intake drawer** — single optical, bi-temporal image pair, and Sentinel-1 SAR dual-pol.
- **Bi-temporal change detection** (ChangeFormer T4) with grounded polygons for vegetation gain, built-up expansion, and water retention.
- **Multi-session history switcher** with three example scenarios (campus expansion, algal bloom, mangrove audit).
- **AOI Magic Wand**, draggable split-swipe comparison, and an interactive mini-calendar for orbit-epoch switching.

---

## 🚀 Quick start

```bash
# 3D landing
cd llm/frontend
python serve.py 3000
# open http://localhost:3000

# Dashboard
cd llm/satquery-frontend-dashboard
python serve.py 3001
# open http://localhost:3001
```

> A local web server is required — browsers block WebGL/WASM/`fetch()` over `file://`.

---

## 🛠️ Tech stack

- **Graphics**: WebGL2 + PlayCanvas Engine
- **Maps**: Leaflet 1.9.4 + Esri World Imagery
- **Styling**: TailwindCSS 3.4 with a custom liquid-glassmorphism layer
- **Animation**: GSAP + custom offscreen-canvas workers
- **3D formats**: Binary glTF (`.glb`) and Basis Universal Texture Compression
- **Audio**: HTML5 Web Audio API (multi-channel stems + UI SFX)
- **WebAssembly**: Google Basis Universal Transcoder

---

## 💡 Local LLM note

The dashboard talks to a vision-language model over a local HTTP endpoint. We packaged a thin **GeoChat API** wrapper (`geochat_api.py`) that runs the model on your own machine — useful for offline demos and judge runs. See the script for setup.

---

## 🙏 Acknowledgements & thanks

A lot of this project stands on the shoulders of generous, brilliant work from the open-source community. We want to give proper credit and say thank you:

- **🎨 The 3D landing page is heavily inspired by [EDOLUS](https://edolus.com/).**
  We studied their site, learned from the way they choreograph the 3D scene, the camera walks, the audio stems, the lighting, the shader work, the whole vibe. Our landing page is a learning tribute to that — thank you, EDOLUS team, for the inspiration. If you happen to read this, we really appreciate the work you put into that experience. 🙏

- **🧠 The local LLM is based on [GeoChat](https://github.com/mbzuai-oryx/GeoChat).**
  GeoChat is the multi-modal large-language model for remote-sensing images from MBZUAI's Oryx lab. We use the model locally for captioning, visual question answering, and grounded region reasoning over satellite imagery. Huge thanks to the GeoChat authors — Kuckreja, Danish, Akhtar, et al. — for open-sourcing their work. The earth-observation community is much better off with GeoChat in it. 🌍

- **🛰️ Sentinel-1 SAR / Sentinel-2 optical sample data** — used for evaluation workflows.
- **🗺️ Map tiles** — courtesy of Esri World Imagery.

If we forgot to credit you, please open an issue and we'll add you immediately.

---

## 👤 Authors

- **Trushendar Reddy** — frontend & integration lead (this repo).
- **Prakash** — original dashboard layout and glassmorphism system ([Sat-Query-Frontend-Dashboard-](https://github.com/PrakashMB-1213/Sat-Query-Frontend-Dashboard-)).

Built with ❤️ for SIH 2026.
