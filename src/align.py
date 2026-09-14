import cv2
from .haar_5pt import Haar5ptDetector,align_face_5pt
def main():
 cap=cv2.VideoCapture(0); det=Haar5ptDetector()
 while True:
  ok,f=cap.read()
  if not ok: break
  faces=det.detect(f)
  if faces:
   a,_=align_face_5pt(f,faces[0].kps); cv2.imshow('aligned 112x112',a)
  cv2.imshow('camera',f)
  if cv2.waitKey(1)&255==ord('q'):break
 cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
