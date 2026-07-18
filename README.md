# Vision Assist — On-Device Walking Companion

A **fully on-device** visual assistant for blind and low-vision pedestrians.  
It detects obstacles, reads signs, remembers places in AR world space, and answers spoken questions — with **no cloud inference**.

Built for the Gemma / on-device AI hackathon. Primary demo target: **iPhone Pro (LiDAR)** via the iOS app in `ios/GemmaVision/`.

---

## What it does

| Capability | How |
|---|---|
| Obstacle alerts | Rule-based voice (“person ahead, 2 meters”) — **never LLM** |
| Object detection | YOLO26n (CoreML) + LiDAR depth |
| Walls / doors / path clear | ARKit Scene Geometry + mesh classification |
| Sign reading | Apple Vision OCR |
| Destination goals | “Looking for the restroom” → alert when sign matches |
| Spatial memory | ARKit world coordinates (not GPS) — “Where’s my backpack?” |
| Recall | “Did I pass the restroom?” — episode log + synonyms |
| Empty seats | YOLO person↔chair IoU → `occupied` (rule-based answer) |
| Open Q&A | Gemma 4 E2B on-device (LiteRT-LM) + detector hints + camera JPEG |

**Hard rule:** safety alerts are deterministic templates only. Gemma is used for user Q&A (and a few routed intents stay rule-based for reliability).

---

## Architecture

```
Camera / ARSession
    ├─ YOLO26n (ANE)     → objects, distances, seat occupancy
    ├─ LiDAR sceneDepth  → meters for boxes + spatial memory
    ├─ ARKit mesh (opt.) → wall / door / window / floor / fork
    └─ Vision OCR        → signs → goals + episode memory
                │
                ▼
         Rule engine (alerts, goals, recall, seats, guide-back)
                │
    PTT (Speech) → router → rules  OR  Gemma 4 E2B (GPU) + hints + image
                │
                ▼
              TTS (AVSpeech) — p0 alerts suppressed during protected answers
```

### Why split detection and LLM?

- **Alerts must be fast and trustworthy** → rules + sensors.
- **Open questions need language** → on-device Gemma with a compact SceneState-style JSON (`detector_hints`) so answers stay grounded.
- **ARKit Scene Geometry** is not a custom segmentation model: Apple builds a LiDAR mesh and labels faces (wall/door/floor…). We sample nearby faces and turn them into spoken structure alerts. Toggle **Mesh OFF** by default to save heat; turn **Mesh ON** when you need walls / path clear.

---

## Tech stack

### iOS (demo)

| Layer | Stack |
|---|---|
| UI | SwiftUI |
| Capture | ARKit (`ARSession`) on Pro; `AVCapture` fallback |
| Detection | YOLO26n `.mlpackage` via Vision / CoreML |
| Structure | ARKit `meshWithClassification` + `sceneDepth` |
| OCR | Vision `VNRecognizeTextRequest` |
| LLM | [Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) via LiteRT-LM (LLM on Metal GPU, vision encoder on CPU) |
| Speech | Speech framework (STT) + `AVSpeechSynthesizer` (TTS) |

### Mac prototype (optional)

Python 3.11, YOLO (MPS), Florence-2 OCR, Ollama Gemma — same SceneState / rule-alert philosophy. See `IMPLEMENTATION.md` and `src/`.

---

## Quick start (iOS)

Full steps: [`ios/SETUP.md`](ios/SETUP.md).

1. Clone and open the Xcode project:
   ```bash
   git clone https://github.com/Yoondoyoung/gemmaHackathon.git
   cd gemmaHackathon
   open ios/GemmaVision/GemmaVision.xcodeproj
   ```
2. **Download the Gemma model** (~2.4GB — not in git):
   ```bash
   # from repo root, with venv if you use the Mac tooling
   python -c "from huggingface_hub import hf_hub_download; \
     hf_hub_download('litert-community/gemma-4-E2B-it-litert-lm', \
     'gemma-4-E2B-it.litertlm', local_dir='ios/GemmaVision/GemmaVision')"
   ```
   Add `gemma-4-E2B-it.litertlm` to the Xcode target (Copy items if needed).
3. Set your **Signing Team**, plug in an iPhone, build & run (⌘R).
4. Allow Camera / Microphone / Speech Recognition.

**Requirements:** iPhone with LiDAR (12 Pro or newer) for mesh + best depth. YOLO and Q&A still work without mesh.

Demo cue sheet: [`DEMO.md`](DEMO.md).

---

## Question routing (PTT)

When you hold **Hold to talk** and release:

1. **Find / guide back** — “Where’s my backpack?” → AR world pose  
2. **Recall (rules)** — “Did I pass the restroom?” → episode / sign memory  
3. **Empty seats** — YOLO `occupied`  
4. **Goal** — “I want to go to the restroom” → watch for matching signs  
5. **Else** — Gemma + `detector_hints` + JPEG  

---

## Repository layout

```
ios/GemmaVision/     # Primary demo app (Xcode)
src/                 # Mac / Python prototype
prompts/             # System prompt (Mac)
IMPLEMENTATION.md    # Implementation contract (hackathon rules)
DEMO.md              # 3-minute demo script
ios/SETUP.md         # iOS build & device checklist
```

---

## Design principles

1. **On-device only** — works in airplane mode once models are on the phone.  
2. **LLM off the safety path** — obstacles and structure alerts are templates.  
3. **Grounded Q&A** — pass verified detections (`label`, `pos`, `dist`, `occupied`, signs, structures), not raw speculation.  
4. **Demo-friendly load** — YOLO ~15 FPS; mesh off until needed; mesh scan limited to ~8 m around the camera.  

---

## Team / hackathon notes

- SceneState JSON field contract: see `IMPLEMENTATION.md` §2 (Mac). iOS uses a reduced snapshot for Gemma.  
- Do not put Gemma on the alert path.  
- Prefer fixing reliability with rules (seats, recall, goals) over prompt-only tweaks when the failure mode is known.

---

## License / models

- App code: hackathon project (see repository).  
- Gemma 4 E2B LiteRT-LM: Apache 2.0 (Litert Community on Hugging Face).  
- YOLO26n weights: included as CoreML package for the iOS target.
