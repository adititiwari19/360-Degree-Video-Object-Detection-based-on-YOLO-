
'''
Given a directory, for each frame (.jpg, .png) inside, project the 360 video frame and its bounding boxes (.txt) into several cropped frames 
with their corresponding bounding boxes according to the projection orientations defined by the projection centers (my_centersX and my_centersY) 
and view angle (my_view_angle). Outputs (frames and bounding boxes) are saved into a new dictionary.

Example usage: python projectAllBB3y.py sample
File requirements: Directory named sample in the current directory, with each image.png and image.txt inside 

Reference 
https://github.com/phananh1010/360-object-detection-annotation/blob/master/annotator.py
https://github.com/phananh1010/360-object-detection-annotation/blob/master/VideoRecorder.py

'''

from msilib.schema import Directory
import VideoRecorder as vr
import cv2
import numpy as np
import os
import sys

IMAGE_ENDINGS = (".png", ".jpg", ".jpeg")
def getAllImages(path):
    images_filenames = []
    for f in os.listdir(path):
        if any(f.endswith(ending) for ending in IMAGE_ENDINGS):
            images_filenames.append(f)

    return images_filenames

def getXTheta(sphereW, x):
		theta_x = (2.* x - 1.) / sphereW - 1.
		return theta_x * np.pi

def getYTheta(sphereH, y):
    theta_y = 0.5 - (y - 0.5) / sphereH
    return theta_y * np.pi

def findXYInWindow(x_angles, y_angles, targetX, targetY):
    distance = sys.maxsize * 2 + 1
    resX = -1
    resY = -1

    for y in range(len(x_angles)):
        for x in range(len(x_angles[0])):
            currentX = x_angles[y][x]
            currentY = y_angles[y][x]
            if abs(targetX - currentX) + abs(targetY - currentY) < distance:
                resX = x
                resY = y
                distance = abs(targetX - currentX) + abs(targetY - currentY)

    return (resX, resY) if distance < 0.01 else (-1, -1)

def transformBB(image_filename, x_angles, y_angles, image_directory):
    label_filename = image_filename[0 : image_filename.index('.')] + '.txt'
    f = open(image_directory + '\\' + label_filename, "r")
    lines = f.readlines()
    windowBoundingBox = []

    for line in lines:
        line = line[0:len(line)-1]
        splitted = line.split(' ')

        type_num = int(splitted[0])
        p1 = (float(splitted[1]), float(splitted[2]))
        p2 = (float(splitted[3]), float(splitted[4]))
        p3 = (float(splitted[5]), float(splitted[6]))
        p4 = (float(splitted[7]), float(splitted[8]))

        allIn = True
        windowCoordinates = [type_num]
        for (targetX, targetY) in [p1, p2, p3, p4]:
            (windowX, windowY) = findXYInWindow(x_angles, y_angles, targetX, targetY)
            windowCoordinates.append(windowX)
            windowCoordinates.append(windowY)
            if windowX == -1:
                allIn = False
                break
        
        if allIn == False:
            continue

        windowBoundingBox.append(windowCoordinates)
    
    return windowBoundingBox

def fourPointsToYolo(windowBoundingBox, window_resolution):
    yoloBoundingBox = []
    
    for windowCoordinates in windowBoundingBox:
        yoloCoordinates = [windowCoordinates[0]]

        x1, y1 = windowCoordinates[1], windowCoordinates[2]
        x2, y2 = windowCoordinates[3], windowCoordinates[4]
        x3, y3 = windowCoordinates[5], windowCoordinates[6]
        x4, y4 = windowCoordinates[7], windowCoordinates[8]

        xtl = 0.5*(x1 + x3)
        xbr = 0.5*(x2 + x4)
        ytl = 0.5*(y1 + y2)
        ybr = 0.5*(y3 + y4)

        x = (xtl + 0.5*(xbr - xtl)) / window_resolution
        y = (ytl + 0.5*(ybr - ytl)) / window_resolution
        w = (xbr - xtl) / window_resolution
        h = (ybr - ytl) / window_resolution

        yoloCoordinates.append(x)
        yoloCoordinates.append(y)
        yoloCoordinates.append(w)
        yoloCoordinates.append(h)
        yoloBoundingBox.append(yoloCoordinates)
        print(yoloCoordinates)
    
    return yoloBoundingBox


def writeBBToTxt(txt_name, windowBoundingBox):
    with open(txt_name, 'w') as f:
        for windowCoordinates in windowBoundingBox:
            f.write(' '.join([str(c) for c in windowCoordinates]) + '\n')
        

def projectXY(image_filename, x, y, view_angle, window_resolution, image_directory, projections_directory, camera):
    src = cv2.imread(image_directory + '\\' + image_filename)

    sphereH, sphereW, _ = map(int, src.shape) # Height and width of the 360 image in Cartesian Coordinate

    # Height and width of the 360 image in Polar Coordinate
    theta_x = getXTheta(sphereW, x)
    theta_y = getYTheta(sphereH, y)

    ir = vr.ImageRecorder(sphereW, sphereH, view_angle, window_resolution)
    x_angles, y_angles = ir._direct_camera(theta_x, theta_y)
    projected_image = ir.catch(theta_x, theta_y, src.copy())
    projected_image[projected_image>255.] = 255.
    projected_image[projected_image<0.] = 0.
    projected_image = projected_image.astype(np.uint8)

    output_name = str(camera) + '_' + image_filename[0 : image_filename.index('.')] + '.png'
    cv2.imwrite(projections_directory + '\\' + output_name, projected_image)

    txt_name = str(camera) + '_' + image_filename[0 : image_filename.index('.')] + '.txt'
    
    windowBoundingBox = transformBB(image_filename, x_angles, y_angles, image_directory)
    yoloBoundingBox = fourPointsToYolo(windowBoundingBox, window_resolution)

    writeBBToTxt(projections_directory + '\\' + txt_name, yoloBoundingBox)

def projectAll(input_name, view_angle, window_resolution, image_directory, projections_directory):
    centersX = [0, 1280, 2560]
    centersY = [640, 1280]

    camera = 1
    for cx in centersX:
        for cy in centersY:
            projectXY(input_name, cx, cy, view_angle, window_resolution, image_directory, projections_directory, camera)
            camera += 1

def main():
    args = sys.argv

    image_directory = args[1]
    image_filenames = getAllImages(image_directory)
    
    projections_directory = 'Projected_' + image_directory
    os.mkdir(projections_directory)

    for image_filename in image_filenames:
        print('Processing: ' + image_filename)
        projectAll(image_filename, 120, 1280, image_directory, projections_directory)
    
if __name__ == "__main__":
    main()