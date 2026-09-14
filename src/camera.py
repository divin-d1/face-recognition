import cv2
def main():
 cap=cv2.VideoCapture(0)
 if not cap.isOpened(): raise RuntimeError('Camera not opened')
 print("Camera test. Press q to quit.")
 while True:
  ok,frame=cap.read()
  if not ok: break
  cv2.imshow('Camera Test',frame)
  if cv2.waitKey(1)&255==ord('q'): break
 cap.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
