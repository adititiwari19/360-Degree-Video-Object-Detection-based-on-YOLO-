
'''
Project the 360 video frame (.jpg, .png) into several cropped frame according to the projection orientations 
defined by the points (my_centersX and my_centersY) and view angle (my_view_angle). Outputs are saved into a new dictionary.

Example usage: python project32.py sample.png
File requirements: sample.png, VideoRecorder.py are in the current directory

Reference 
https://github.com/phananh1010/360-object-detection-annotation/blob/master/annotator.py
https://github.com/phananh1010/360-object-detection-annotation/blob/master/VideoRecorder.py

'''

import VideoRecorder as vr
import cv2
import numpy as np
import os
import sys

my_centersX = [0, 1280, 2560]
my_centersY = [640, 1280]

'''
view_angle = 120: Each projected frame takes: 1280 x 1280 pixels from the original 360 video frame
view_angle = 104.3: Each projected frame takes: 1114 x 1114 pixels from the original 360 video frame
view_angle = 90: Each projected frame takes: 960 x 960 pixels from the original 360 video frame
view_angle = 65.5: Each projected frame takes: 700 x 700 pixels from the original 360 video frame
'''
my_view_angle = 120

def getXTheta(sphereW, x):
		theta_x = (2.* x - 1.) / sphereW - 1.
		return theta_x * np.pi

def getYTheta(sphereH, y):
    theta_y = 0.5 - (y - 0.5) / sphereH
    return theta_y * np.pi

def projectXY(input_name, x, y, view_angle, window_resolution, projections_directory, camera):
    src = cv2.imread(input_name)

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

    output_name = str(camera) + '_' + input_name[0 : input_name.index('.')] + '.png'
    cv2.imwrite(projections_directory + '\\' + output_name, projected_image)

def projectAll(input_name, centersX, centersY, view_angle, window_resolution = 1280):
    projections_directory = 'Projected_' + input_name[0 : input_name.index('.')]
    os.mkdir(projections_directory)

    camera = 1
    for cx in centersX:
        for cy in centersY:
            projectXY(input_name, cx, cy, view_angle, window_resolution, projections_directory, camera)
            camera += 1

def main():
    args = sys.argv

    input_name = args[1]
    projectAll(input_name, my_centersX, my_centersY, my_view_angle)
    
if __name__ == "__main__":
    main()


