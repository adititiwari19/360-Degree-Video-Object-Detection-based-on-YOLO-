
'''
Given a 360 video frame (.jpg, .png) and its bounding boxes labeled in polar coordinate (.txt), annotate its bounding boxes
and save the annotated frame in the current directory. This tool will also annotate the confidence value if existed in the txt file.
The label name and label colors are defined according to the custom firefighting dataset.

Example usage: python annotate3_f.py sample.png
File requirements: sample.png, sample.txt are in the current directory

Reference 
https://github.com/phananh1010/360-object-detection-annotation/blob/master/annotator.py

'''

import cv2
import os
import sys
import numpy as np

COLORS = [
    (255, 192, 203),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (51, 153, 255),
    (255, 255, 0),
    (204, 0, 204),
    (255, 0, 0),
    (255, 192, 203),
    (255, 192, 203),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 153),
    (255, 128, 0),
    (0, 255, 255),
    (255, 255, 255),
    (0, 0, 255),
    (255, 128, 0),
    (255, 192, 203),
    (0, 0, 0)
]

NAMES = ['Yolo BB 0 Based Indexing', 'Firefighter', 'Civilian', 'Ladder', 'Fire', 'Window', 'Oxygen Tank', 'Door', 'Gas Tank', 'Fire Truck', 'Firefighter Helmet',
        'Structural Damage', 'Civilian Car', 'Trees', 'Water Hose', 'Building', 'Fence', 'Stairs', 'Water', 'Firefighter Mask', 'Smoke']

def angle_coords(sphereW, sphereH, point):	#turns radians into screen coordinates for the equirectangular image
    return (int(sphereW*(point[0]+np.pi) / (2 * np.pi)), int(sphereH*(np.pi/2 - point[1]) / np.pi))

def annotate(image_filename):
    label_filename = image_filename[0 : image_filename.index('.')] + '.txt'
    output_name = image_filename[0 : image_filename.index('.')] + '_anno' + '.png'

    src = cv2.imread(image_filename)
    anno_image = src.copy()

    sphereH, sphereW, _ = map(int, src.shape) # Height and width of the 360 image in Cartesian Coordinate

    f = open(label_filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line[0:len(line)-1]
        splitted = line.split(' ')

        type_num = int(splitted[0])
        p1 = (float(splitted[1]), float(splitted[2]))
        p2 = (float(splitted[3]), float(splitted[4]))
        p3 = (float(splitted[5]), float(splitted[6]))
        p4 = (float(splitted[7]), float(splitted[8]))

        fs = 0.8
        tn = 2
        label = str(NAMES[type_num])
        if len(splitted) > 9:
            label = label + str("{0:.2g}".format(float(splitted[9])))

        w, h = cv2.getTextSize(label, 0, fontScale=fs, thickness=tn)[0]
        lt = (angle_coords(sphereW, sphereH, p1)[0], angle_coords(sphereW, sphereH, p1)[1])
        text_rb = (angle_coords(sphereW, sphereH, p1)[0] + w, angle_coords(sphereW, sphereH, p1)[1] + h)
        cv2.rectangle(anno_image, lt, text_rb, COLORS[type_num], -1, cv2.LINE_AA)  # filled
        cv2.putText(anno_image, label, (lt[0], lt[1] + h), 0, fs, (255, 255, 255),
                            thickness=tn, lineType=cv2.LINE_AA)

        for (pt_a, pt_b) in ((p1, p2), (p1, p3), (p2, p4), (p3, p4)):
            if abs(pt_a[0] - pt_b[0]) > np.pi: #wraparound, requires two lines
                pt_a, pt_b = max(pt_a, pt_b, key=lambda x:x[0]), min(pt_a, pt_b, key=lambda x:x[0])
                cv2.line(anno_image, angle_coords(sphereW, sphereH, pt_a), angle_coords(sphereW, sphereH, (pt_b[0] + 2*np.pi, pt_b[1])), COLORS[type_num], 3)
                cv2.line(anno_image, angle_coords(sphereW, sphereH, pt_b), angle_coords(sphereW, sphereH, (pt_a[0] - 2*np.pi, pt_a[1])), COLORS[type_num], 3)
            else:
                cv2.line(anno_image, angle_coords(sphereW, sphereH, pt_a), angle_coords(sphereW, sphereH, pt_b), COLORS[type_num], 3)

    cv2.imwrite(output_name, anno_image)

def main():
    args = sys.argv

    image_filename = args[1]
    annotate(image_filename)
        
    
if __name__ == "__main__":
    main()