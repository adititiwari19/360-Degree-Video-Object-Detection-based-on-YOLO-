## Apply Projections to 360 Video Frames

### project32.py
Project the 360 video frame (.jpg, .png) into several cropped frame according to the projection orientations defined by the points (my_centersX and my_centersY) and view angle (my_view_angle). Outputs are saved into a new dictionary. \
Usage: `python project32.py sample.png`

### projectAllBB3y
Given a directory, for each frame (.jpg, .png) inside, project the 360 video frame and its bounding boxes (.txt) into several cropped frames with their corresponding bounding boxes according to the projection orientations defined by the projection centers (my_centersX and my_centersY) and view angle (my_view_angle). Outputs (frames and bounding boxes) are saved into a new dictionary. \
Usage: `python projectAllBB3y.py sample`