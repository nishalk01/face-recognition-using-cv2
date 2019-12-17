import cv2
import numpy
from fun import ret_rect
imag=cv2.imread('-1x-1.jpg')
w,h,_=imag.shape
img,faces,coordinate=ret_rect(imag)
if (faces==1):
 img=cv2.resize(img,(400,500))
 img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 recognizer = cv2.face.LBPHFaceRecognizer_create()
 recognizer.read('trainer.yml')
 id, confidence = recognizer.predict(img)
 if(id==0):
     name="elon_musk"
 else :
     name="gal_gadot"

 if(confidence>=40):
     print("i donno who u are")
 else:
     for x,y,w,h in coordinate:
      #
      cv2.rectangle(imag,(x,y),(x+w,y+h),(0,255,0),2)
      fontScale = (w * h) / (500 * 500)
      cv2.putText(imag,name,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,fontScale,(255,255,0),4)
      cv2.imwrite('recognized.jpg',imag)
else:
    print("no faces found")
