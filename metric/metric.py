
'''
Given all the test samples (.png, .jpg, .jpeg) inside the image directory, all the corresponding real labels (.txt) inside 
the real directory, and all the corresponding predicted labels (.txt) inside pred directory (can be specified by the user),
return the precision, recall, and F1 score for each class, people average, and frequent class average.

It assumes the real labels to be in polar coordinates and predicted labels in YOLO format. The IOU is calculated based on the
original shape of the bounding boxes. The default threshold is set to be 0.5 and can be specified by the user.

Example usage: python metric.py pred 0.5
File requirements: image, real, and pred directories are in the current directory

Reference 
https://github.com/phananh1010/360-object-detection-annotation/blob/master/annotator.py

'''

import cv2
import os
import sys
import numpy as np
from shapely.geometry import Polygon

CLASSES = ['Yolo BB 0 Based Indexing', 'Firefighter', 'Civilian', 'Ladder', 'Fire', 'Window', 'Oxygen Tank', 'Door', 'Gas Tank', 'Fire Truck', 'Firefighter Helmet',
        'Structural Damage', 'Civilian Car', 'Trees', 'Water Hose', 'Building', 'Fence', 'Stairs', 'Water', 'Firefighter Mask', 'Smoke']

IMAGE_ENDINGS = (".png", ".jpg", ".jpeg")
def getAllImages(path):
    images_filenames = []
    for f in os.listdir(path):
        if any(f.endswith(ending) for ending in IMAGE_ENDINGS):
            images_filenames.append(f)
    return images_filenames

def angle_coords(sphereW, sphereH, point):	#turns radians into screen coordinates for the equirectangular image
    return (int(sphereW*(point[0]+np.pi) / (2 * np.pi)), int(sphereH*(np.pi/2 - point[1]) / np.pi))

def read_bb_polar(bb_filename):
    bbs = []

    if os.path.exists(bb_filename) == False:
        return bbs

    f = open(bb_filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line[0:len(line)-1]
        splitted = line.split(' ')

        bb = [int(splitted[0]), float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4]), float(splitted[5]), float(splitted[6]), float(splitted[7]), float(splitted[8])]
        if len(splitted) >= 10:
            bb.append(float(splitted[9]))
        bbs.append(bb)
    return bbs

def bb_polar2xyxyxyxy(bb_polar, sphereW, sphereH):
    bbs_xyxyxyxy = []

    for bb in bb_polar:
        p1 = (bb[1], bb[2])
        p2 = (bb[3], bb[4])
        p3 = (bb[5], bb[6])
        p4 = (bb[7], bb[8])

        # TODO: Check for images across the boundary 
        two_line = False
        for (pt_a, pt_b) in ((p1, p2), (p1, p3), (p2, p4), (p3, p4)):
            if abs(pt_a[0] - pt_b[0]) > np.pi: 
                two_line = True
        if two_line:
            continue

        x1, y1 = angle_coords(sphereW, sphereH, p1)[0], angle_coords(sphereW, sphereH, p1)[1]
        x2, y2 = angle_coords(sphereW, sphereH, p2)[0], angle_coords(sphereW, sphereH, p2)[1]
        x3, y3 = angle_coords(sphereW, sphereH, p3)[0], angle_coords(sphereW, sphereH, p3)[1]
        x4, y4 = angle_coords(sphereW, sphereH, p4)[0], angle_coords(sphereW, sphereH, p4)[1]

        x1, x2, x3, x4 = 1. * x1 / sphereW, 1. * x2 / sphereW, 1. * x3 / sphereW, 1. * x4 / sphereW
        y1, y2, y3, y4 = 1. * y1 / sphereH, 1. * y2 / sphereH, 1. * y3 / sphereH, 1. * y4 / sphereH

        bb_xyxyxyxy = [bb[0], x1, y1, x2, y2, x3, y3, x4, y4]
        if len(bb) >= 10:
            bb_xyxyxyxy.append(bb[9])
        
        bbs_xyxyxyxy.append(bb_xyxyxyxy)
    return bbs_xyxyxyxy

def read_bb_yolo(bb_filename):
    bbs = []

    if os.path.exists(bb_filename) == False:
        return bbs

    f = open(bb_filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line[0:len(line)-1]
        splitted = line.split(' ')

        bb = [int(splitted[0]), float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])]
        if len(splitted) >= 6:
            bb.append(float(splitted[5]))
        bbs.append(bb)
    return bbs

def bb_xywh2xyxyxyxy(bb_xywh):
    bbs_xyxyxyxy = []

    for bb in bb_xywh:
        x1, y1 = bb[1] - 0.5 * bb[3], bb[2] - 0.5 * bb[4]
        x2, y2 = bb[1] + 0.5 * bb[3], bb[2] - 0.5 * bb[4]
        x3, y3 = bb[1] - 0.5 * bb[3], bb[2] + 0.5 * bb[4]
        x4, y4 = bb[1] + 0.5 * bb[3], bb[2] + 0.5 * bb[4]

        bb_xyxyxyxy = [bb[0], x1, y1, x2, y2, x3, y3, x4, y4]
        if len(bb) >= 6:
            bb_xyxyxyxy.append(bb[5])
        
        bbs_xyxyxyxy.append(bb_xyxyxyxy)
    return bbs_xyxyxyxy

def bb_scale(bbs, width, height):
    bbs_scaled = []

    for bb in bbs:
        x1, x2, x3, x4 = bb[1] * width, bb[3] * width, bb[5] * width, bb[7] * width
        y1, y2, y3, y4 = bb[2] * height, bb[4] * height, bb[6] * height, bb[8] * height

        bb_scaled = [bb[0], x1, y1, x2, y2, x3, y3, x4, y4]
        if len(bb) >= 10:
            bb_scaled.append(bb[9])
        
        bbs_scaled.append(bb_scaled)
    return bbs_scaled

def single_confusion_matrix(bb_real_8, bb_pred_8, classes, iou_threshold=0.6):
    tp = []
    real_p = []
    pred_p = []

    for class_idx in range(1, len(classes)):
        bb_real_class = [bb for bb in bb_real_8 if bb[0] == class_idx]
        bb_pred_class = [bb for bb in bb_pred_8 if bb[0] == class_idx]

        correct_real = []
        correct_pred = []
        for j in range(len(bb_real_class)):
            for i in range(len(bb_pred_class)):
                if i in correct_pred:
                    continue
                
                bbr = bb_real_class[j]
                bbp = bb_pred_class[i]
                
                pol_r = [[bbr[1], bbr[2]], [bbr[3], bbr[4]], [bbr[7], bbr[8]], [bbr[5], bbr[6]]]
                pol_p = [[bbp[1], bbp[2]], [bbp[3], bbp[4]], [bbp[7], bbp[8]], [bbp[5], bbp[6]]]

                polygon_r_shape = Polygon(pol_r)
                polygon_p_shape = Polygon(pol_p)

                intersection = polygon_r_shape.intersection(polygon_p_shape).area
                union = polygon_r_shape.union(polygon_p_shape).area
                iou = intersection / union 

                if iou > iou_threshold:
                    correct_real.append(j)
                    correct_pred.append(i)
                    break
        
        tp.append(len(correct_pred))
        real_p.append(len(bb_real_class))
        pred_p.append(len(bb_pred_class))
        assert (len(correct_real) == len(correct_pred))

    return tp, real_p, pred_p
        
def main():
    args = sys.argv

    pred_directory = args[1]
    threshold = float(args[2])

    image_directory = 'image'
    real_directory = 'real'
    
    image_filenames = getAllImages(image_directory)
    
    tps, real_ps, pred_ps = np.zeros(len(CLASSES) - 1), np.zeros(len(CLASSES) - 1), np.zeros(len(CLASSES) - 1)
    for image_filename in image_filenames:
        #print('Processing: ' + image_filename)

        image_path = image_directory + '\\' + image_filename
        bb_real_path = real_directory + '\\' + image_filename[0 : image_filename.index('.')] + '.txt'
        bb_pred_path = pred_directory + '\\' + image_filename[0 : image_filename.index('.')] + '.txt'

        src = cv2.imread(image_path)
        sphereH, sphereW, _ = map(int, src.shape)

        bb_real_raw = read_bb_polar(bb_real_path)
        bb_real_8 = bb_polar2xyxyxyxy(bb_real_raw, sphereW, sphereH)
        bb_real_8 = bb_scale(bb_real_8, sphereW, sphereH)

        bb_pred_raw = read_bb_yolo(bb_pred_path)
        bb_pred_8 = bb_xywh2xyxyxyxy(bb_pred_raw)
        bb_pred_8 = bb_scale(bb_pred_8, sphereW, sphereH)

        tp, real_p, pred_p = single_confusion_matrix(bb_real_8, bb_pred_8, CLASSES, threshold)
        tps = tps + np.array(tp)
        real_ps = real_ps + np.array(real_p)
        pred_ps = pred_ps + np.array(pred_p)
    
    for class_idx in range(1, len(CLASSES)):
        print(str(class_idx) + ' ' + CLASSES[class_idx])
        if pred_ps[class_idx - 1] == 0:
            print('Precision: Divided by 0')
        else:
            print('Precision: ' + str(tps[class_idx - 1]) + '/' + str(pred_ps[class_idx - 1]) + '=' + str(tps[class_idx - 1] / pred_ps[class_idx - 1]))
        
        if real_ps[class_idx - 1] == 0:
            print('Recall: Divided by 0')
        else:
            print('Recall: ' + str(tps[class_idx - 1]) + '/' + str(real_ps[class_idx - 1]) + '=' + str(tps[class_idx - 1] / real_ps[class_idx - 1]) )
        
        if pred_ps[class_idx - 1] != 0 and real_ps[class_idx - 1] != 0:
            print('F1 Score: ' + str( 2 * tps[class_idx - 1] / (2 * tps[class_idx - 1] + pred_ps[class_idx - 1] + real_ps[class_idx - 1]) ) )

        print()

    prople_idx = [0, 1]
    frequent_class_idx = [0, 1, 2, 4, 5, 8, 9]

    tp_p, pred_ps_p, real_ps_p = 0, 0, 0
    tp_fre, pred_ps_fre, real_ps_fre = 0, 0, 0
    
    for i in prople_idx:
        tp_p += tps[i]
        pred_ps_p += pred_ps[i]
        real_ps_p += real_ps[i]
    
    for i in frequent_class_idx:
        tp_fre += tps[i]
        pred_ps_fre += pred_ps[i]
        real_ps_fre += real_ps[i]
    
    print('People Avg Precision: ' + str( tp_p / pred_ps_p)) 
    print('People Avg Recall: ' + str( tp_p / real_ps_p))
    print('People Avg F1 Score: ' + str( 2 * tp_p / (2 * tp_p + pred_ps_p + real_ps_p) ) )

    print('Freq Class Avg Precision: ' + str( tp_fre / pred_ps_fre)) 
    print('Freq Class Avg Recall: ' + str( tp_fre / real_ps_fre))
    print('Freq Class Avg F1 Score: ' + str( 2 * tp_fre / (2 * tp_fre + pred_ps_fre + real_ps_fre) ) )
    
    
if __name__ == "__main__":
    main()










        


