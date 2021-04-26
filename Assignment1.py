# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: Sai Sudharshan Prerepa
"""

#......IMPORT .........
import argparse
import matplotlib.pyplot as MatPlot 
import cv2 as OPENCV
import os as OperatingSystem
import numpy as NUMPY


def GetImageAverage(ImageData):
    GrayAverage = OPENCV.mean(ImageData)
    return GrayAverage[0]/2

#Custom Median Filter for different sizes of filter
def Image_MedianFilter(FilterSize, InputImage):
   
    K = FilterSize//2
    GrayValues = []
    for R in range(0,len(InputImage)):
        for C in range(0,len(InputImage[0])):
            XStart = R - K
            YStart = C - K
            XEnd = R + K
            YEnd = C + K
            for X in range(XStart, XEnd + 1):
                for Y in range(YStart, YEnd + 1):
                    if (X >= 0 and X < len(InputImage)):
                        if (Y>=0 and Y< len(InputImage[0])):
                            GrayValues.append(InputImage[X][Y])
                    else:
                        GrayValues.append(0)
            GrayValues.sort()
            InputImage[R][C] = GrayValues[len(GrayValues)//2]
            GrayValues = []
    return InputImage

def GetNBHValidPixels(i, j, BO_Median):
    R = BO_Median.shape[0]
    C = BO_Median.shape[1]
    Kernel_Pixels = []
    White = 255
    if i-1 >= 0:
        if j-1 >= 0:
            if BO_Median[i-1][j-1] != White:
                Kernel_Pixels.append(BO_Median[i-1][j-1])
        if BO_Median[i-1][j] != White:
            Kernel_Pixels.append(BO_Median[i-1][j])
        if j+1 < C:
            if BO_Median[i-1][j+1] != White:
                Kernel_Pixels.append(BO_Median[i-1][j+1])
    if j-1 >= 0:
        if BO_Median[i][j-1] != White:
            Kernel_Pixels.append(BO_Median[i][j-1])
    if j+1 < C:
        if BO_Median[i][j+1] != White:
            Kernel_Pixels.append(BO_Median[i][j+1])
    if i+1 < R:
        if j-1 >= 0:
            if BO_Median[i+1][j-1] != White:
                Kernel_Pixels.append(BO_Median[i+1][j-1])
        if BO_Median[i+1][j] != White:
            Kernel_Pixels.append(BO_Median[i+1][j])
        if j+1 < C:
            if BO_Median[i+1][j+1] != White:
                Kernel_Pixels.append(BO_Median[i+1][j+1])
    Kernel_Pixels_U = sorted(list(set(Kernel_Pixels)))
    return Kernel_Pixels_U

def Task1(ImageFileName, Output):
    ImageData = OPENCV.imread(ImageFileName, 1)
    ImageInGray = OPENCV.cvtColor(ImageData, OPENCV.COLOR_BGR2GRAY)

    Itr = []
    Thresh = []
    Itr.append(0)
    IGrayAvg = GetImageAverage(ImageData)
    Thresh.append(IGrayAvg)
    MatPlot.imshow(ImageInGray)

    C = True

    LeftGrayCount = 0
    RightGrayCount = 0
    LeftGrayTotal = 0
    RightGrayTotal = 0
    LeftGrayMean = 0
    RightGrayMean = 0
    GrayAvgPresent = IGrayAvg

    Rows = ImageInGray.shape[0]
    Cols = ImageInGray.shape[1]
    i=0
    while(C == True):
        r = 0
        while (r < Rows):
            c = 0
            while (c < Cols):
                if ImageInGray[r][c] > GrayAvgPresent:
                    RightGrayTotal = RightGrayTotal + ImageInGray[r][c]
                    RightGrayCount = RightGrayCount + 1
                else:
                    LeftGrayTotal = LeftGrayTotal + ImageInGray[r][c]
                    LeftGrayCount = LeftGrayCount + 1
                c = c + 1
            r = r + 1
        if LeftGrayCount == 0:
            LeftGrayMean = 0
        else:
            LeftGrayMean = LeftGrayTotal/LeftGrayCount
        if RightGrayCount == 0:
            RightGrayMean
        else:
            RightGrayMean = RightGrayTotal/RightGrayCount
        GrayAvgPrevious = GrayAvgPresent
        GrayAvgPresent = (LeftGrayMean + RightGrayMean) /2
        if  abs(GrayAvgPresent - GrayAvgPrevious) <= 0.03:
            C = False
        i = i + 1
        Itr.append(i)
        Thresh.append(GrayAvgPresent)
        LeftGrayTotal = 0
        LeftGrayCount = 0
        RightGrayTotal = 0
        RightGrayCount = 0

    ImageBinary = ImageInGray.copy()
    r = 0
    White = 255
    Black = 0
    while (r < Rows):
        c = 0
        while (c < Cols):
            if ImageInGray[r][c] > GrayAvgPresent:
                ImageBinary[r][c] = Black
            else:
                ImageBinary[r][c] = White
            c = c + 1
        r = r + 1
    BinaryOutput = ImageFileName[:-4] + '_Task1.png'

    ImageWithGrains = OPENCV.cvtColor(ImageBinary, OPENCV.COLOR_BGR2RGB)
    MatPlot.imshow(ImageWithGrains)
    TitleText = "Threshold Value = " + str(round(GrayAvgPresent, 2))
    MatPlot.title(TitleText)
    MatPlot.xticks([])
    MatPlot.yticks([])

    MatPlot.savefig(Output+'/'+BinaryOutput)
    MatPlot.show()
    
    MatPlot.xticks(Itr)
    MatPlot.xlabel("Iterations")
    MatPlot.ylabel("Threshold values")
    MatPlot.plot(Itr, Thresh)
    MatPlot.show()
    return ImageBinary

#TASK 2
def Task2(ImageBinary, ImageFileName, Output):
    BO_Median = Image_MedianFilter(5, ImageBinary)

    Seed_Val = 1
    Rows = BO_Median.shape[0]
    Cols = BO_Median.shape[1]
    for R in range(Rows):
        for C in range(Cols):
            if BO_Median[R][C] == 0:
                List_Val = GetNBHValidPixels(R, C, BO_Median)
                if len(List_Val) > 0:
                    if max(List_Val) > 0:
                        if min(List_Val) > 0:
                            BO_Median[R][C] = List_Val[0]
                        else:
                            BO_Median[R][C] = List_Val[1]
                    else:
                        BO_Median[R][C] = Seed_Val
                        Seed_Val = Seed_Val + 1
                else:
                    BO_Median[R][C] = 255

    RiceGrains = []
    White = 255
    for R in range(Rows):
        for C in range(Cols):  
            if BO_Median[R][C] < White:
                RiceGrains.append(BO_Median[R][C])

    GrainOutput = ImageFileName[:-4] + '_Task2.png'

    Diff = 0
    while Diff == 0:
        Diff = 1
        for R in range(Rows):
            for C in range(Cols):
                if BO_Median[R][C] < White:
                    List_Val = GetNBHValidPixels(R, C, BO_Median)
                    if len(List_Val) > 0:
                        min_class = min(List_Val)
                        if BO_Median[R][C] != min_class:
                            Diff = 0
                            BO_Median[R][C] = min_class


    RiceGrains = []
    for R in range(Rows):
        for C in range(Cols):  
            if BO_Median[R][C] < White:
                RiceGrains.append(BO_Median[R][C])
    FinalRiceGrains = list(set(RiceGrains))          
    ImageWithGrains = OPENCV.cvtColor(BO_Median, OPENCV.COLOR_BGR2RGB)
    MatPlot.imshow(ImageWithGrains)
    TitleText = "Number of RIce Kernels = " + str(len(FinalRiceGrains))
    MatPlot.title(TitleText)
    MatPlot.xticks([])
    MatPlot.yticks([])

    MatPlot.savefig(Output+'/'+GrainOutput)
    MatPlot.show()
    return BO_Median, FinalRiceGrains
    
#TASK 3
def Task3(BO_Median, filename, FinalRiceGrains, MinimumArea, Output):
    Rows = BO_Median.shape[0]
    Cols = BO_Median.shape[1]
    
    RiceGrainAreas = []
    for G in FinalRiceGrains:
        A = 0
        R = 0
        while R < Rows:
            C = 0
            while C < Cols: 
                if BO_Median[R][C] == G:
                    A = A + 1
                C = C  + 1
            R = R + 1
        RiceGrainAreas.append(A)
    AreaThresh = MinimumArea
    
    DamagedGrains = 0
    DamagedRiceGrains = []
    i = 0
    for RGA in RiceGrainAreas:
        if RGA < AreaThresh:
            DamagedRiceGrains.append(RGA)
            DamagedGrains = DamagedGrains + 1
            R = 0
            while R < Rows:
                C = 0
                while C < Cols:
                    if BO_Median[R][C] == FinalRiceGrains[i]:
                        BO_Median[R][C] = 255
                    C = C + 1
                R = R + 1
        i = i + 1
    DamagedPercent = (DamagedGrains/len(RiceGrainAreas))*100
    GrainOutput = filename[:-4] + '_Task3.png'
    ImageWithGrains = OPENCV.cvtColor(BO_Median, OPENCV.COLOR_BGR2RGB)
    MatPlot.imshow(ImageWithGrains)
    TitleText = "Percentage of Damaged Kernels = " + str(round(DamagedPercent, 3))
    MatPlot.title(TitleText)
    MatPlot.xticks([])
    MatPlot.yticks([])
    MatPlot.savefig(Output+'/'+GrainOutput)
    MatPlot.show()

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()
ImageFilenName = args.input_filename
MinimumArea = args.min_area
Output = args.OP_folder

if OperatingSystem.path.isdir('./'+Output) != True: 
    OperatingSystem.mkdir(Output)

ImageBinary = Task1(ImageFileName, Output)
BO_Median, FinalRiceGrains = Task2(ImageBinary, ImageFileName, Output)
Task3(BO_Median, ImageFileName, FinalRiceGrains, MinimumArea, Output)

