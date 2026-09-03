# SatQuery 3D Landing (PlayCanvas) 🛰️✨

A standalone cinematic 3D WebGL landing experience that pairs with the SatQuery AI geospatial dashboard.

Built on the **PlayCanvas** engine with GSAP animations, multi-track audio, and a hand-crafted scene of orbital satellites, compute hardware, and shader-driven lighting.

---

## 🌟 Overview

This is the 3D landing scene for SatQuery AI. It opens with an animated camera walk through a stylized orbit-and-compute environment, then transitions into the analytical dashboard.

---

## 🚀 Quick start

```bash
python serve.py 3000
# open http://localhost:3000
```

A local web server is required — browsers block WebGL/WASM/`fetch()` over `file://`.

---

## 📦 What's included

- **3D models (`.glb`)**: Tesla, ComputeTray, Starlink V2, GDX pods, Diamond V4, MAP, CurvedScreen, triangle3D, AutonomousDeployment
- **Audio stems (`.ogg`)**: melody / instruments / bass / UI SFX (BTNclick2, diamond2, map, rack, computecore2, textchip2, quantum5, UIscreen, AICHIP, intelligent, Scrambletext)
- **HD video backgrounds (`.mp4`)**: Space-compress + Seedance2-0 cinematic loops
- **Universal textures (`.basis`)**: WebAssembly Basis Universal compressed maps (normal, AO, diffuse, reflection, skybox)
- **Engine runtime**: PlayCanvas Engine, GSAP, Basis Transcoder WASM, and the project scene graph (`2509662.json` + `config.json`)

---

## 🙏 Acknowledgements

This 3D landing was **heavily inspired by [EDOLUS](https://edolus.com/)** — we studied their site, the camera walks, the audio stems, the lighting, the shader work, and the overall cinematic feel. Our landing page is a learning tribute to that experience. Thank you, EDOLUS team, for the inspiration. 🙏

---

## 👤 Author

- **Trushendar Reddy** — integration, asset curation, repo merge.
- Built for **Smart India Hackathon 2026**.
