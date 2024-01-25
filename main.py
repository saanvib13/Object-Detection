from Detection import *
import os

def main():
    # videoPath='videos/video2.mp4'
    # videoPath='images/image-1.jpg'
    videoPath=0
    configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    modelPath='frozen_inference_graph.pb'
    classPath='labels.txt'

    detector=Detection(videoPath, configPath, modelPath, classPath)
    detector.onVideo()

if __name__=='__main__':
    main()