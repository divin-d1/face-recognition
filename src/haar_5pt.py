from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2, numpy as np
import mediapipe as mp

@dataclass
class FaceKpsBox:
    x1:int; y1:int; x2:int; y2:int; score:float; kps:np.ndarray

IDX=(33,263,1,61,291)

def _kps_ok(k,min_eye=12.0):
    le,re,no,lm,rm=k; return float(np.linalg.norm(re-le))>=min_eye and lm[1]>no[1] and rm[1]>no[1]

def _bbox(k,padx=.55,padt=.85,padb=1.15):
    xmin,xmax=k[:,0].min(),k[:,0].max(); ymin,ymax=k[:,1].min(),k[:,1].max()
    w=max(1.,xmax-xmin); h=max(1.,ymax-ymin)
    return np.array([xmin-padx*w,ymin-padt*h,xmax+padx*w,ymax+padb*h],np.float32)

def _clip(b,W,H): return np.array([np.clip(b[0],0,W-1),np.clip(b[1],0,H-1),np.clip(b[2],0,W-1),np.clip(b[3],0,H-1)],np.float32)

def _ema(prev,cur,a): return cur.astype(np.float32) if prev is None else (a*prev+(1-a)*cur).astype(np.float32)

def _matrix(k,out=(112,112)):
    dst=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],np.float32)
    if out!=(112,112): dst*=np.array([out[0]/112,out[1]/112],np.float32)
    M,_=cv2.estimateAffinePartial2D(k.astype(np.float32),dst,method=cv2.LMEDS)
    if M is None: raise RuntimeError('Could not estimate 5-point alignment transform')
    return M.astype(np.float32)

def align_face_5pt(frame,kps,out_size=(112,112)):
    M=_matrix(kps,out_size)
    return cv2.warpAffine(frame,M,(int(out_size[0]),int(out_size[1])),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),M

class Haar5ptDetector:
    def __init__(self,min_size=(70,70),smooth_alpha=.80,debug=False):
        self.debug=debug; self.min_size=min_size; self.alpha=smooth_alpha
        self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if self.cascade.empty(): raise RuntimeError('Failed to load Haar cascade')
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
        self.prev_box=None; self.prev_kps=None
    def detect(self,frame,max_faces=1)->List[FaceKpsBox]:
        H,W=frame.shape[:2]; gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=self.cascade.detectMultiScale(gray,1.1,5,minSize=self.min_size)
        if len(faces)==0:return []
        faces=sorted(faces,key=lambda r:r[2]*r[3],reverse=True)[:max_faces]
        out=[]
        for x,y,w,h in faces:
            mx,my=.25*w,.35*h; rx1=max(0,int(x-mx)); ry1=max(0,int(y-my)); rx2=min(W,int(x+w+mx)); ry2=min(H,int(y+h+my))
            roi=frame[ry1:ry2,rx1:rx2]
            res=self.mesh.process(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue
            lm=res.multi_face_landmarks[0].landmark
            k=np.array([[lm[i].x*roi.shape[1]+rx1,lm[i].y*roi.shape[0]+ry1] for i in IDX],np.float32)
            if k[0,0]>k[1,0]: k[[0,1]]=k[[1,0]]
            if k[3,0]>k[4,0]: k[[3,4]]=k[[4,3]]
            if not _kps_ok(k,max(10.,.18*w)): continue
            b=_clip(_bbox(k),W,H); b=_ema(self.prev_box,b,self.alpha); k=_ema(self.prev_kps,k,self.alpha)
            self.prev_box=b; self.prev_kps=k
            out.append(FaceKpsBox(*map(lambda z:int(round(z)),b),1.0,k.astype(np.float32)))
        return out
