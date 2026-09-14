from pathlib import Path
import cv2,numpy as np
from .haar_5pt import Haar5ptDetector,align_face_5pt
from .embed import ArcFaceEmbedderONNX

def main():
    dbp=Path('data/db/face_db.npz')
    if not dbp.exists(): raise RuntimeError('No enrollment DB. Run python -m src.enroll first.')
    d=np.load(dbp,allow_pickle=True); names=list(d.files); mat=np.stack([d[k].astype(np.float32) for k in names])
    emb=ArcFaceEmbedderONNX(); det=Haar5ptDetector(debug=False); cap=cv2.VideoCapture(0); thr=.34
    if not cap.isOpened(): raise RuntimeError('Camera not available')
    print('q quit | +/- threshold | r reload DB')
    while True:
        ok,frame=cap.read()
        if not ok: break
        vis=frame.copy()
        for f in det.detect(frame,max_faces=5):
            aligned,_=align_face_5pt(frame,f.kps); q=emb.embed(aligned); sims=mat@q; i=int(np.argmax(sims)); sim=float(sims[i]); dist=1-sim
            known=dist<=thr; label=names[i] if known else 'Unknown'; color=(0,255,0) if known else (0,0,255)
            cv2.rectangle(vis,(f.x1,f.y1),(f.x2,f.y2),color,2); cv2.putText(vis,f'{label} dist={dist:.3f} sim={sim:.3f}',(f.x1,max(20,f.y1-8)),0,.6,color,2)
        cv2.putText(vis,f'IDs={len(names)} threshold={thr:.2f}',(10,30),0,.7,(255,255,255),2); cv2.imshow('Face Recognition',vis)
        key=cv2.waitKey(1)&255
        if key==ord('q'): break
        if key in (ord('+'),ord('=')): thr=min(1.2,thr+.01)
        elif key==ord('-'): thr=max(.05,thr-.01)
        elif key==ord('r'):
            d=np.load(dbp,allow_pickle=True); names=list(d.files); mat=np.stack([d[k].astype(np.float32) for k in names]); print('reloaded',names)
    cap.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
