# Face Recognition with ArcFace ONNX and 5-Point Alignment

A CPU-only, explainable face recognition pipeline built stage by stage: face
detection, 5-point landmark extraction, similarity-transform alignment,
ArcFace ONNX embedding, enrollment, and threshold-based recognition. Built
for the *Face Recognition with ArcFace ONNX and 5-Point Alignment* practical
(Benax Technologies Ltd / Rwanda Coding Academy).

## Architecture / Pipeline

```
Camera
  -> Haar Cascade Face Detection
  -> MediaPipe FaceMesh 5-Point Landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner)
  -> Similarity-Transform Alignment (rotation + scale + translation -> 112x112)
  -> ArcFace ONNX Embedding (512-D, L2-normalized)
  -> Enrollment (average multiple sample embeddings into one template) / Recognition (cosine distance vs. stored templates + threshold)
  -> Known <name> / Unknown
```

| File | Responsibility |
|---|---|
| `src/app.py` | **Unified live app**: full pipeline (detect + landmarks + align + embed + recognize) in one fullscreen window, with enrollment built in |
| `src/camera.py` | Camera sanity check (open, read, FPS) |
| `src/detect.py` | Haar face bounding-box sanity check |
| `src/landmarks.py` | 5-point landmark sanity check |
| `src/align.py` | Alignment sanity check (112x112 output) |
| `src/embed.py` | ArcFace ONNX embedding + `ArcFaceEmbedderONNX` class |
| `src/haar_5pt.py` | Combined Haar detector + 5-point landmarks + alignment used by enroll/recognize |
| `src/camera_utils.py` | Shared `--camera` CLI argument / device opening helper |
| `src/enroll.py` | Multi-sample enrollment -> averaged, L2-normalized template |
| `src/recognize.py` | Live multi-face recognition against the enrolled database |
| `src/evaluate.py` | Genuine/impostor distance evaluation and threshold suggestion |

## Dependencies

```
opencv-python
numpy
onnxruntime
scipy
tqdm
mediapipe==0.10.21
```

### Why Python 3.11 (not whatever `python3` defaults to)

`mediapipe==0.10.21` is required because it is the last line of releases
that still ships the legacy `mediapipe.solutions.face_mesh` API this project
uses for 5-point landmarks. Newer mediapipe releases (0.10.30+, 1.0.x) have
**removed** `mediapipe.solutions` entirely in favor of a different Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`), and `mediapipe==0.10.21`
itself only ships wheels for Python 3.9-3.12 (no `cp313` wheel), so it
cannot be `pip install`-ed under a Python 3.13 interpreter. If your system
Python is 3.13 (`python3 --version`), build an isolated Python 3.11 with
pyenv rather than changing the pinned dependency:

```bash
# One-time: build dependencies (Debian/Kali; needs sudo)
sudo apt-get update && sudo apt-get install -y \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev tk-dev

# Install pyenv and build Python 3.11 locally (no system Python changes)
git clone --depth 1 https://github.com/pyenv/pyenv.git ~/.pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11.16
```

If your system already has a Python 3.9-3.12 interpreter, skip pyenv and use
it directly instead.

## Environment Setup

```bash
cd face-recognition
~/.pyenv/versions/3.11.16/bin/python3.11 -m venv .venv   # or: python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python init_project.py   # idempotent; creates data/, models/, src/ if missing
```

Verify the interpreter and key imports before continuing:

```bash
python --version          # Python 3.11.x
python -c "import cv2, numpy, onnxruntime, scipy, mediapipe; \
           from mediapipe.solutions import face_mesh; print('all imports OK')"
```

## Model Setup (ArcFace ONNX)

The pipeline needs `models/embedder_arcface.onnx`, the `w600k_r50.onnx`
ResNet-50 ArcFace embedder from InsightFace's `buffalo_l` model pack.

```bash
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

**If the SourceForge mirror is slow or unreachable** (observed in some
network environments), download the identical `w600k_r50.onnx` file
directly from a public Hugging Face mirror instead:

```bash
curl -L -o models/embedder_arcface.onnx \
  "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
```

Both sources serve the exact same InsightFace `buffalo_l/w600k_r50.onnx`
file (only the ArcFace embedder is used here — the other files in
`buffalo_l.zip`, such as the InsightFace face/landmark detectors, are not
needed because this project uses Haar + MediaPipe for detection/landmarks).

Validate the model loads and produces a correct embedding:

```bash
python -m src.embed --camera /dev/video2
```
Expected: a window opens showing `embedding dim: 512`, `norm(before L2):`
around 15-30, and (after a couple of frames) `cos(prev,this):` close to 1.0
for a stationary face.

## External Camera Setup

List connected cameras and their device nodes:

```bash
v4l2-ctl --list-devices
```

Every script accepts `--camera <path-or-index>` (default: `/dev/video2`,
or override via the `FACE_REC_CAMERA` environment variable):

```bash
python -m src.camera --camera /dev/video2
```

OpenCV index `0` is **not guaranteed** to map to a particular `/dev/videoN`
node when multiple cameras are attached — always pass the explicit device
path for the camera you intend to use.

## Quick Start: Unified App

`src/app.py` runs the whole pipeline together in one fullscreen window
(1280x720 by default) with enrollment built in — the fastest way to
actually use the system day to day:

```bash
python -m src.app --camera /dev/video2
```

Keys: `e` enroll a new person (prompts for a name in the terminal), `SPACE`
capture a sample while enrolling, `a` toggle auto-capture, `s` save the
enrollment, `c` cancel enrollment, `r` reload the database, `+`/`-` adjust
the recognition threshold, `f` toggle fullscreen, `q` quit.

The per-stage scripts below remain available for isolated debugging that
matches the PDF's Chapter 1 validation steps (confirm one stage at a time
if something looks wrong in the unified app).

## Validate Each Stage

```bash
python -m src.camera --camera /dev/video2       # q to quit
python -m src.detect --camera /dev/video2        # q to quit
python -m src.landmarks --camera /dev/video2     # q to quit
python -m src.align --camera /dev/video2         # q to quit
python -m src.embed --camera /dev/video2         # q to quit
```

## Enrollment

```bash
python -m src.enroll --camera /dev/video2 --name Divin
```
(Omit `--name` to be prompted interactively.) Position your face in frame,
press `SPACE` to capture a sample (or `a` to toggle auto-capture), collect
at least 5-15 samples across slightly different angles/expressions, then
press `s` to save. Repeat for every person you want the system to know.
Aligned 112x112 crops are saved under `data/enroll/<name>/` for later
evaluation; the averaged, L2-normalized template is stored in
`data/db/face_db.npz` / `data/db/face_db.json`.

## Recognition

```bash
python -m src.recognize --camera /dev/video2 --threshold 0.34
```
`q` quits, `+`/`-` adjust the cosine-distance accept threshold live, `r`
reloads the database from disk. Enrolled faces are labeled with their name
and a green box; unrecognized faces are labeled `Unknown` with a red box.

## Evaluation

```bash
python -m src.evaluate
```
Computes genuine (same-person) vs. impostor (different-person) cosine
distance distributions from the aligned crops saved during enrollment,
prints a threshold sweep (FAR/FRR at each threshold), and suggests a
threshold for a 1% target false-accept rate. Enroll at least two people
with saved crops for a meaningful impostor distribution.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install mediapipe==0.10.21` fails on Python 3.13 | Use the pyenv + Python 3.11 setup above; this pin has no `cp313` wheel. |
| `ImportError: cannot import name 'face_mesh' from 'mediapipe.solutions'` | You're on mediapipe >=0.10.30 or 1.x, which dropped the legacy Solutions API. Reinstall the pinned `mediapipe==0.10.21` under Python 3.11. |
| Camera not opening | Confirm the device with `v4l2-ctl --list-devices`, pass the correct `--camera /dev/videoN`, and make sure no other app holds the camera. |
| `RuntimeError: No enrollment DB` in recognize | Run `python -m src.enroll` for at least one person first. |
| Flat/incorrect embeddings, meaningless distances | Confirm `models/embedder_arcface.onnx` is the real ArcFace model (~166 MB), not a placeholder; re-run `python -m src.embed` to sanity-check. |
| Impostor distances empty in evaluate | Enroll at least two different people with saved crops. |
| Flickering identity labels | Increase smoothing / hold time, or loosen lighting/framing during enrollment for more representative samples. |

## Expected Output

- `python -m src.embed`: `embedding dim: 512`, `norm(before L2): ~15-30`, unit-norm after L2 normalization.
- `python -m src.enroll`: `data/db/face_db.npz` and `data/db/face_db.json` created/updated with one 512-D template per identity.
- `python -m src.recognize`: enrolled faces labeled with name + green box + `dist=`/`sim=`; unknown faces labeled `Unknown` + red box.
- `python -m src.evaluate`: genuine distances clustered low, impostor distances clustered high, and a suggested threshold near the observed separation point.

## What Not to Commit

`.gitignore` already excludes:
- `.venv/`, `__pycache__/`, `*.pyc`
- `models/*.onnx` (the ArcFace model — downloaded during setup, not tracked)
- `data/enroll/*` and `data/db/*` (personal biometric images and embeddings)

Only `.gitkeep` placeholders are tracked for `data/enroll/` and `data/db/`
so the expected project structure is preserved without committing anyone's
face data.
