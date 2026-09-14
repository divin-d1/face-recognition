from pathlib import Path
import cv2,numpy as np
from .embed import ArcFaceEmbedderONNX
def main():
 people=[p for p in Path('data/enroll').iterdir() if p.is_dir()] if Path('data/enroll').exists() else []
 emb=ArcFaceEmbedderONNX(); per={}
 for p in people:
  xs=[]
  for f in sorted(p.glob('*.jpg')):
   im=cv2.imread(str(f))
   if im is not None and im.shape[:2]==(112,112): xs.append(emb.embed(im))
  if len(xs)>=2: per[p.name]=xs
 gen=[]; imp=[]
 names=list(per)
 for n in names:
  for i in range(len(per[n])):
   for j in range(i+1,len(per[n])): gen.append(1-float(np.dot(per[n][i],per[n][j])))
 for i in range(len(names)):
  for j in range(i+1,len(names)):
   for a in per[names[i]]:
    for b in per[names[j]]: imp.append(1-float(np.dot(a,b)))
 print('genuine:',len(gen),'impostor:',len(imp));
 if gen: print('genuine mean',np.mean(gen))
 if imp: print('impostor mean',np.mean(imp))
if __name__=='__main__':main()
