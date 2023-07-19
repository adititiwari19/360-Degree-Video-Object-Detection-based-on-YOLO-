import cv2
import os
import sys
import numpy as np

def histogram_equalization_yuv(src):
    '''
    Convert the input image to YUV color space and apply the Histogram Equalization to the Y channel.
    
    Parameters:
    src (np.ndarray): input image

    Returns:
    numpy.ndarray: image applied with Histogram Equalization
    '''

    src_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    src_yuv[:, :, 0] = cv2.equalizeHist(src_yuv[:, :, 0])
    dst = cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR)
    
    return dst

def main():
    args = sys.argv

    try:
        input_name = args[1]
    except Exception as e:
        print(e)
        print('Usage error: should be python heyuv.py image_path!')
        return

    if not os.path.exists(input_name):
        print('The image file does not exist!')
        return
    
    output_name = input_name[0 : input_name.index('.')] + '_yuv.jpg'

    src = cv2.imread(input_name)
    dst = histogram_equalization_yuv(src)
    
    cv2.imwrite(output_name, dst)

if __name__ == "__main__":
    main()