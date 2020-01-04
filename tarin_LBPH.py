import numpy as np
import cv2

label=np.load('y.npy', allow_pickle=True)
print(label)
ppl=np.load('ppl.npy', allow_pickle=True)
print(ppl[0])
#show all the images of  the dataset
for hm in ppl:
 hm=cv2.resize(hm,(400,500))
 cv2.imshow('d',hm)
 cv2.waitKey(100)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(ppl, label)
recognizer.write('trainer.yml')
