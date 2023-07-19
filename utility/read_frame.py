import cv2
import os
import sys
import numpy as np

def read_frame(video_path, requested_frame_number):
    '''
    Extract a frame of frame number specified from the a video, and save it into the local directory.
    
    Parameters:
    video_path (string): File path to the video to extract frame from
    requested_frame_number (int): Frame number of the frame to extract
    '''

    if not os.path.exists(video_path):
        print('The video file does not exist!')
        return

    output_name = video_path[0 : video_path.index('.')-1] + '_' + str(requested_frame_number) + '.jpg'
    cap = cv2.VideoCapture(video_path)

    if requested_frame_number > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        print('The requested frame number should be less than the total number of frames!')
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, requested_frame_number)
    ret, frame = cap.read()
    cv2.imwrite(output_name, frame)

def main():
    args = sys.argv

    try:
        video_path = args[1]
        requested_frame_number = int(args[2])
    except Exception as e:
        print(e)
        print('Usage error: should be python read_frame.py video_path requested_frame_number!')
        return

    read_frame(video_path, requested_frame_number)
if __name__ == "__main__":
    main()