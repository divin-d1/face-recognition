from __future__ import annotations
import cv2, numpy as np, onnxruntime as ort
class ArcFaceEmbedderONNX:
    def __init__(self,model_path='models/embedder_arcface.onnx',input_size=(112,112),debug=False):
        self.w,self.h=input_size; self.sess=ort.InferenceSession(model_path,providers=['CPUExecutionProvider'])
        self.in_name=self.sess.get_inputs()[0].name; self.out_name=self.sess.get_outputs()[0].name
        if debug: print('input',self.sess.get_inputs()[0].shape,'output',self.sess.get_outputs()[0].shape)
    def embed(self,img):
        if img.shape[:2]!=(self.h,self.w): img=cv2.resize(img,(self.w,self.h))
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32); x=np.transpose((rgb-127.5)/128.0,(2,0,1))[None]
        v=self.sess.run([self.out_name],{self.in_name:x.astype(np.float32)})[0].reshape(-1).astype(np.float32)
        return v/(np.linalg.norm(v)+1e-12)
