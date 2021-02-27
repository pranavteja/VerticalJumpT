import argparse

import cv2
import matplotlib.image as mpimg
import copy
import numpy as np
import pandas as pd

import bodypose_model
import util
import body
import torch
from scipy.ndimage.filters import gaussian_filter
import math

import sys, os
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter


def Vertical_Jumps(video_path,mass):
    
    weight_path = os.path.join(sys.path[0], 'weights/body_pose_model.pth')
    body_estimation = body.Body(model_path  = weight_path)
    body_estimation.createModel()

    input_path = img_path = os.path.join(sys.path[0], video_path)
    df = pd.DataFrame({'frame_no':[],'RA_x':[],'RA_y':[],'LA_x':[],'LA_y':[]})

    input = cv2.VideoCapture(input_path) 

    if input.isOpened(): 
        frame_width  = input.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        frame_height = input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = input.get(cv2.CAP_PROP_FPS)
        frame_count = input.get(cv2.CAP_PROP_FRAME_COUNT)

    size = (int(frame_width), int(frame_height) )

    i=0   
    while(True): 
        input.set(1,i)
        ret, frame = input.read() 

        if ret == True:  
            candidate, subset = body_estimation(frame)

            if subset.shape[0] >0 :
                index = int(subset[0][13])
                if index != -1:
                    lax, lay = candidate[index][0:2]
                else :
                    lax = -1
                    lay = -1

                index = int(subset[0][10])
                if index != -1:
                    rax, ray = candidate[index][0:2]
                else :
                    rax = -1
                    ray = -1

                df = df.append(pd.DataFrame({'frame_no':[i],'RA_x':[rax],'RA_y':[ray],'LA_x':[lax],'LA_y':[lay]}))

        else: 
            break


        i+= int(fps/6)


    input.release() 

    cv2.destroyAllWindows() 
    
    data = df.iloc[:,0:5].reset_index(drop=True, inplace=False).values

    j = 1 
    while j < data[0].shape[0] :
        i=0
        while(i<data.shape[0]):
            if (data[i][j] == -1) :
                count = 0
                while(data[i+count][j] == -1) :
                    count+=1
                if(i>0):
                    data[i][j] = int((data[i-1][j] + data[i+count][j])/2)
                else :
                    data[i][j] = data[i+count][j]
            i+=1
        j+=1

    div = 10
    spl = splrep(data[:,0],((data[:,2]+data[:,4])/2),k=3)
    x = np.linspace(0, data[-1,0], int(data[-1,0]*div)+1)
    y = splev(x, spl)
    
    jumps = ((y - np.min(y))/(np.max(y)-np.min(y))) < 0.6
    
    m = np.mean(y[jumps==0])
    
    yd1 = np.diff(y,n=1)
    xd1 = x[:-1]
    
    yd1_norm = ((yd1 - np.min(yd1))/(np.max(yd1)-np.min(yd1)))
    
    peaks = yd1_norm
    peaks[peaks>0.8] = 1
    peaks[peaks<0.2] = -1
    peaks[(peaks!=1)*(peaks!=-1)] = 0
    
    pairs = np.array([[0,0]])
    i = 0
    while(i < peaks.shape[0]):
        if peaks[i] == -1 :
            a = i
            while i< peaks.shape[0] and peaks[i] == -1 :
                i+=1
            b = i-1
            c = a + np.argmin(yd1[a:b])

            while i< peaks.shape[0] and peaks[i] == 0 :
                i+=1

            if i< peaks.shape[0] and peaks[i] == 1 :
                a  = i
                while i< peaks.shape[0] and peaks[i] == 1 :
                    i+=1
                b = i-1
                d = a + np.argmax(yd1[a:b])
                pairs = np.append(pairs,[[c,d]],axis = 0)

            i-=1
        i+=1

    pairs = pairs[1:]
    
    f1 = np.array([[0,0]])
    for i in pairs:
        a = int(i[0] - (y[i[0]] - m)/yd1[i[0]])
        b = int(i[1] + (m-y[i[1]])/yd1[i[1]])
        f1 = np.append(f1,[[x[a],x[b]]],axis = 0)
    f1 = f1[1:]

    f = f1
    
    t = np.array([0])
    for i in f :
        t = np.append(t,(i[1]-i[0])/fps)
    t = t[1:]
    
    Average_jump_height = 0
    Jump_height = np.zeros(t.shape[0])
    Average_energy_spent = 0
    Energy_spent = np.zeros(t.shape[0])
    No_jumps = 0

    for i in range(0,t.shape[0]):
        Jump_height[i] = 9.8*t[i]*t[i]/8
        Energy_spent[i] = mass*9.8*Jump_height[i] 

    Average_jump_height = np.mean(Jump_height)
    Average_energy_spent = np.mean(Energy_spent)
    No_jumps = t.shape[0]
    
    return No_jumps,Average_jump_height,Average_energy_spent
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jump keypoint detection')
    parser.add_argument("--video_file", help="Input Video")
    parser.add_argument("--mass", default=60, help="Input mass")
    args = parser.parse_args()
    Vertical_Jumps(args.video_file,args.mass)
else:
    print("the file is being called indirectly")