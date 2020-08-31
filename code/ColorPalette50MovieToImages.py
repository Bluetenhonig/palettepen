# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:00:34 2020

@author: Anonym
"""

# load modules 
import os
import cv2
import numpy as np


# declare variables 
KPS = 5 # Keyframes Per Second
VIDEO_PATH = r"D:\thesis\videos" 
VIDEO_FILE = "tagesschau.mp4"
OUTPUT_PATH =  r"D:\thesis\videos\frames"  
EXTENSION = "jpg"

#%%

if __name__ == '__main__':
    
    # Read video file 
    os.chdir(VIDEO_PATH)
    vidcap = cv2.VideoCapture(VIDEO_FILE)
    success,image = vidcap.read()

#%%   
    # Get frames in video file    
    seconds = KPS
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds

#%%     
    # Save frames to PC
    os.chdir(OUTPUT_PATH)
    while success:
        frameId = int(round(vidcap.get(1))) 
        success, image = vidcap.read()
    
        if frameId % multiplier == 0:
            cv2.imwrite(f"frame%d.{EXTENSION}" % frameId, image)
    
    vidcap.release()
    print("Complete")


