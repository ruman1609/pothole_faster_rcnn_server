# Server for Mobile in [Pothole Faster R-CNN Detection](https://github.com/ruman1609/pothole-faster-rcnn-detection)
This server using Django

You can see the [requirements](requirements.txt) and install it in your virtual environment of your server.

If you are using different image size, don't forget change [models folder](model_tf/models) and [utils folder](model_tf/utils) with your own framework folder.

## Notice
1. Don't forget to put your created .h5 format weight files (VGG16, ResNet50, and MobileNetV2) in [model_tf folder](model_tf/)  
   the data tree will be like this for [model_tf folder](model_tf/)
   ```bash
   .
   │
   ├───...   
   │
   ├───model_tf
   │   ├──faster_rcnn_mobilenet_v2_weights.h5
   │   ├──faster_rcnn_resnet50_weights.h5
   │   ├──faster_rcnn_vgg16_weights.h5
   │   ├───models
   │   │   ├──faster_rcnn.py
   │   │   ├──rpn_mobilenet_v2.py
   │   │   ├──rpn_resnet50.py
   │   │   ├──rpn_vgg16.py
   │   │   ├──__init__.py
   │   │   └──__pycache__
   │   │      └───...
   │   │
   │   │
   │   └───utils
   │       ├──bbox_utils.py
   │       ├──data_utils.py
   │       ├──drawing_utils.py
   │       ├──eval_utils.py
   │       ├──io_utils.py
   │       ├──train_utils.py
   │       ├──__init__.py
   │       └───__pycache__
   │           └───...
   └───...
   ```
2. Don't delete media folder, because it is the place where images are uploaded and stored.
