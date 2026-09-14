# Face Recognition with ArcFace ONNX and 5-Point Alignment

Pipeline from the provided study material:

Camera -> Haar Face Detection -> MediaPipe 5-Point Landmarks -> Similarity Alignment (112x112) -> ArcFace ONNX 512-D Embedding -> L2 Normalization -> Cosine Matching -> Threshold -> Known/Unknown.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python init_project.py
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

## Validate

```bash
python -m src.camera
python -m src.detect
python -m src.landmarks
python -m src.align
```

## Enroll multiple identities

```bash
python -m src.enroll
```
Enter a name, capture samples with SPACE or press `a` for auto capture, then `s` to save. Repeat for at least two people.

## Recognize

```bash
python -m src.recognize
```
`q` quits, `+/-` adjusts the distance threshold, `r` reloads the database.

## Evaluation

```bash
python -m src.evaluate
```

Do not commit the ONNX model or biometric enrollment images to GitHub. The repository should contain the source, README, requirements, and project structure; the model is downloaded during setup.
