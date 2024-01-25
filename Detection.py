import cv2
import numpy as np
import time
np.random.seed(111)
class Detection:
    def __init__(self, videoPath, configPath, modelPath, classPath):
        self.videoPath=videoPath
        self.configPath=configPath
        self.modelPath=modelPath
        self.classPath=classPath

        self.model_setup=cv2.dnn.DetectionModel(self.modelPath,self.configPath)
        self.model_setup.setInputSize(320,320)
        self.model_setup.setInputScale(1.0/127.5)   # scaling the rgb values
        self.model_setup.setInputMean((127.5,127.5,127.5)) #0-255  255/2 =127.5, mean values of rgb, usd for preprocessing
        self.model_setup.setInputSwapRB(True) 

        self.readClasses()


    def readClasses(self):
        with open(self.classPath, 'r') as f:
            self.classList=f.read().split('\n')
            self.classList.insert(0,'__Background__')
            print(self.classList)

            self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classList),3))

    def onVideo(self):
        vid=cv2.VideoCapture(self.videoPath)
        if vid.isOpened()==False:
            print('File cannot be opened')
            return 
        
        (success,image)=vid.read()
        while success:
            image=cv2.resize(image,(420,420))
            classIndex, confidence, bboxs=self.model_setup.detect(image,confThreshold=0.4)
            bboxs=list(bboxs)
            confidence=list(np.array(confidence).reshape(1,-1)[0])

            confidence=list(map(float, confidence))

            #NMS
            bboxId=cv2.dnn.NMSBoxes(bboxs, confidence, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxId) !=0:
                for i in range(0,len(bboxId)):
                    bbox=bboxs[np.squeeze(bboxId[i])]
                    classConfidence=confidence[np.squeeze(bboxId[i])]
                    classLabelId=np.squeeze(classIndex[np.squeeze(bboxId[i])])
                    classLabel=self.classList[classLabelId]
                    classColor=[int(c) for c in self.colorList[classLabelId]]

                    displayText='{}:{:.2f}'.format(classLabel,classConfidence)

                    x,y,w,h=bbox
                    cv2.rectangle(image,(x,y),(x+w,y+h),color=classColor,thickness=1)
                    cv2.putText(image,displayText,(x+5,y+20),cv2.FONT_HERSHEY_PLAIN,2,classColor,2)


                    linewidth=min(int(w*0.3), int(h*0.3))
                    cv2.line(image,(x,y),(x+linewidth,y),classColor,thickness=4)
                    cv2.line(image,(x,y),(x,y+linewidth),classColor,thickness=4)

                    cv2.line(image,(x+w,y),(x+w-linewidth,y),classColor,thickness=4)
                    cv2.line(image,(x+w,y),(x+w,y+linewidth),classColor,thickness=4)

                    cv2.line(image,(x,y+h),(x+linewidth,y+h),classColor,thickness=4)
                    cv2.line(image,(x,y+h),(x,y+h-linewidth),classColor,thickness=4)

                    cv2.line(image,(x+w,y+h),(x+w-linewidth,y+h),classColor,thickness=4)
                    cv2.line(image,(x+w,y+h),(x+w,y+h-linewidth),classColor,thickness=4)

            cv2.imshow('Result',image)
            key=cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break

            (success,image)=vid.read()
        cv2.destroyAllWindows()




