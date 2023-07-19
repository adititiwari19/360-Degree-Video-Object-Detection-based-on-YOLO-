## Annotate Frames with Bounding Boxes

### annotate3_f.py
Given a 360 video frame (.jpg, .png) and its bounding boxes labeled in polar coordinate (.txt), annotate its bounding boxes and save the annotated frame in the current directory. This tool will also annotate the confidence value if existed in the txt file. The label name and label colors are defined according to the custom firefighting dataset. \
Usage: `python annotate3_f.py sample.png`

### annotate3_y.py
Given a 360 video frame (.jpg, .png) and its bounding boxes labeled in polar coordinate (.txt), annotate its bounding boxes and save the annotated frame in the current directory. This tool will also annotate the confidence value if existed in the txt file. The label name and label colors are defined according to the COCO dataset. \
Usage: `python annotate3_y.py sample.png`