import cv2
import numpy as np
import os
cascade_file="haarcascade_frontalface_default.xml"
cascade=cv2.CascadeClassifier(cascade_file)
ppl=[]
label=[]
def ret_rect(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5,minSize = (24, 24))
    coordinate=faces
    if(faces==()):
         print("No faces where found deleted image")
         roi_color=[]
         faces=0
    else:
      for x,y,w,h in faces:
       roi_color = image[y:y+h, x:x+w]
      faces=1
    return roi_color,faces,coordinate



def dataset_prep():
    path='/home/nishal/learn/data'
    dir=os.listdir(path)

    for img in dir:
        print(img)
        class_num = dir.index(img)
        print(class_num)
        p=os.path.join(path,img)
        i=0
        for data in os.listdir(p):
            pathe_image=os.path.join(p,data)
            image=cv2.imread(pathe_image)
            face,faces,_=ret_rect(image)
            i=i+1
            if(faces==1):
                name=[str(img)+str(i)+'.jpg',i]
                face=cv2.resize(face,(400,500))
                face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                ppl.append(face)
                label.append(class_num)
                cv2.imwrite(os.path.join(p,name[0]),face)
                os.remove(pathe_image)
            else:
                os.remove(pathe_image)




dataset_prep()
np.save('ppl',ppl)
np.save('y',label)
print(label)
