"""
Unified live face recognition app: camera -> Haar detection -> 5-point
landmarks -> alignment (112x112) -> ArcFace ONNX embedding -> recognition,
all in one fullscreen window, with enrollment built in.

The individual stage scripts (src/camera.py, src/detect.py, src/landmarks.py,
src/align.py, src/embed.py, src/enroll.py, src/recognize.py) still exist for
the PDF's per-stage validation/debugging steps; this app is the combined,
demo-ready program that runs the whole pipeline together.

Run:
    python -m src.app --camera /dev/video2

Keys:
    q       quit
    e       start enrolling a new person (prompts for a name in the terminal)
    SPACE   capture an enrollment sample (only while enrolling)
    a       toggle auto-capture (only while enrolling)
    s       save the enrollment (only while enrolling; needs >=5 samples)
    c       cancel enrollment and return to recognition
    r       reload the recognition database from disk
    +/-     adjust the recognition threshold
    f       toggle fullscreen
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera_utils import add_camera_arg, open_camera
from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX

DB_NPZ = Path('data/db/face_db.npz')
DB_JSON = Path('data/db/face_db.json')

WIN = 'Face Recognition'


def load_db():
    db = {}
    if DB_NPZ.exists():
        d = np.load(DB_NPZ, allow_pickle=True)
        db = {k: d[k].astype(np.float32) for k in d.files}
    names = sorted(db)
    mat = np.stack([db[n] for n in names]) if names else None
    return db, names, mat


def save_db(db: dict):
    DB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DB_NPZ, **{k: v.astype(np.float32) for k, v in db.items()})
    DB_JSON.write_text(json.dumps(
        {'names': sorted(db), 'embedding_dim': 512}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Unified live face recognition app.")
    add_camera_arg(parser)
    parser.add_argument('--model', default='models/embedder_arcface.onnx')
    parser.add_argument('--width', type=int, default=1280, help="Requested camera capture width.")
    parser.add_argument('--height', type=int, default=720, help="Requested camera capture height.")
    parser.add_argument('--threshold', type=float, default=.34,
                         help="Cosine-distance accept threshold (lower = stricter).")
    parser.add_argument('--no-fullscreen', action='store_true', help="Start in a normal window instead.")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    det = Haar5ptDetector(debug=False)
    embedder = ArcFaceEmbedderONNX(model_path=args.model)

    db, names, mat = load_db()
    thr = args.threshold
    fullscreen = not args.no_fullscreen

    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    if fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    enrolling = False
    enroll_name = None
    enroll_samples = []
    enroll_auto = False
    enroll_last = 0.0

    print("q quit | e enroll | SPACE capture | a auto-capture | s save | "
          "c cancel | r reload | +/- threshold | f fullscreen")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        H, W = vis.shape[:2]

        faces = det.detect(frame, max_faces=1 if enrolling else 5)
        first_aligned = None

        for idx, f in enumerate(faces):
            for x, y in f.kps.astype(int):
                cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)

            aligned, _ = align_face_5pt(frame, f.kps)
            if idx == 0:
                first_aligned = aligned
            v = embedder.embed(aligned)

            if enrolling:
                color = (255, 200, 0)
                label = f'ENROLLING {enroll_name} ({len(enroll_samples)} samples)'
            elif mat is not None:
                sims = mat @ v
                i = int(np.argmax(sims))
                sim = float(sims[i])
                dist = 1 - sim
                known = dist <= thr
                label = f'{names[i]} dist={dist:.3f} sim={sim:.3f}' if known else f'Unknown dist={dist:.3f}'
                color = (0, 255, 0) if known else (0, 0, 255)
            else:
                label = 'Unknown (no enrolled identities)'
                color = (0, 0, 255)

            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(vis, label, (f.x1, max(24, f.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if first_aligned is not None:
            thumb = cv2.resize(first_aligned, (160, 160))
            vis[10:170, W - 170:W - 10] = thumb
            cv2.rectangle(vis, (W - 170, 10), (W - 10, 170), (255, 255, 255), 1)
            cv2.putText(vis, 'aligned 112x112', (W - 170, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        header = f"IDs={len(names)} thr={thr:.2f}"
        if enrolling:
            header += f"  |  ENROLLING: {enroll_name}"
        cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(WIN, vis)
        key = cv2.waitKey(1) & 0xFF

        now = time.time()
        if enrolling and enroll_auto and first_aligned is not None and now - enroll_last > .25:
            enroll_samples.append(embedder.embed(first_aligned))
            enroll_last = now

        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                   cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('r'):
            db, names, mat = load_db()
            print('reloaded', names)
        elif key in (ord('+'), ord('=')):
            thr = min(1.2, thr + .01)
        elif key == ord('-'):
            thr = max(.05, thr - .01)
        elif key == ord('e') and not enrolling:
            enroll_name = input('Enter name to enroll: ').strip()
            if enroll_name:
                enrolling = True
                enroll_samples = []
                enroll_auto = False
                print(f"Enrolling {enroll_name}: SPACE=capture, a=auto, s=save, c=cancel")
        elif key == ord('c') and enrolling:
            enrolling = False
            enroll_name = None
            enroll_samples = []
            print('Enrollment cancelled.')
        elif key == ord('a') and enrolling:
            enroll_auto = not enroll_auto
        elif key == ord(' ') and enrolling:
            if first_aligned is not None:
                enroll_samples.append(embedder.embed(first_aligned))
                print('captured', len(enroll_samples))
        elif key == ord('s') and enrolling:
            if len(enroll_samples) < 5:
                print('Need at least 5 samples first.')
                continue
            v = np.mean(np.stack(enroll_samples), axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            db[enroll_name] = v.astype(np.float32)
            save_db(db)
            print(f'Saved {enroll_name}. Identities: {len(db)}')
            db, names, mat = load_db()
            enrolling = False
            enroll_name = None
            enroll_samples = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
