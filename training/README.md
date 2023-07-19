## Train a YOLOv3 model based on labeled firefighting data
- Download the folder `camreas` from the Box, and put that in the directory where `yolov3` lives.
- Download `cameras.yaml` and put that inside the `yolov3` directory.
- Run `python train.py --img 640 --batch 16 --epochs 5 --data cameras.yaml --weights yolov3.pt`.
- Modify batch and image size accordingly.