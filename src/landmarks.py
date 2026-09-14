import cv2,numpy as np,mediapipe as mp
IDX=[33,263,1,61,291]
def main():
 cap=cv2.VideoCapture(0); mesh=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)
 while True:
  ok,f=cap.read()
  if not ok: break
  h,w=f.shape[:2]; r=mesh.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
  if r.multi_face_landmarks:
   lm=r.multi_face_landmarks[0].landmark
   for i in IDX: cv2.circle(f,(int(lm[i].x*w),int(lm[i].y*h)),4,(0,255,0),-1)
  cv2.imshow('5pt Landmarks',f)
  if cv2.waitKey(1)&255==ord('q'): break
 cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
