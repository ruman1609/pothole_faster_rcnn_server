from django.shortcuts import render, get_object_or_404 as get_404
from django.http import JsonResponse
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import Prediction
from .serializers import PredictionSerializer

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import numpy as np, tensorflow as tf, random, os

from model_tf.utils import train_utils, bbox_utils
from model_tf.models import faster_rcnn

# backbone = "vgg16"
# backbone = "mobilenet_v2"
backbone = "resnet50"

hyper_params = train_utils.get_hyper_params(backbone)
labels = ["bg", "pothole"]
hyper_params["total_labels"] = len(labels)
anchors = bbox_utils.generate_anchors(hyper_params)

if (backbone == "vgg16"):
    from model_tf.models.rpn_vgg16 import get_rpn_model
elif (backbone == "mobilenet_v2"):
    from model_tf.models.rpn_mobilenet_v2 import get_rpn_model
elif (backbone == "resnet50"):
    from model_tf.models.rpn_resnet50 import get_rpn_model

load_path = os.path.join(settings.MODEL_TF_ROOT, f"faster_rcnn_{backbone}_weights.h5")
rpn_model, feature_extractor = get_rpn_model(hyper_params)
model = faster_rcnn.get_model_frcnn(feature_extractor, rpn_model, anchors, hyper_params, mode="test")
model.load_weights(load_path)


# Create your views here.
def index(request):
    return render(request, "dashboard.html")

class PredictionAPI(APIView):
    __result_path = ""
    __result_id = -1
    __result_object = ""

    def __giveBB(self, filename, image, bboxes, label_indices, probs):
        font = font_manager.FontProperties(family='sans-serif', weight='bold')
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, 16)

        labels = ["bg", "pothole"]
        PredictionAPI.__result_object = ""
        colors = []
        for i in range(len(label_indices)):
            colors.append([255, random.randint(0, 255), random.randint(0, 255), 255])
        draw = ImageDraw.Draw(image)
        for index, bbox in enumerate(bboxes):
            y1, x1, y2, x2 = tf.split(bbox, 4)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            label_index = int(label_indices[index])
            color = tuple(colors[label_index])
            label_text = f"{labels[label_index]} {(probs[index] * 100):.1f}%"
            PredictionAPI.__result_object += f"{label_text}\n"
            draw.text((x1 + 4, y1 + 2), str(index + 1), fill=color, font=font)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        path = os.path.join(settings.MEDIA_ROOT, "results")
        if not os.path.isdir(path):
            os.mkdir(path)

        image.save(os.path.join(path, filename))
        PredictionAPI.__result_path = f"/media/results/{filename}"


    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            ima = Image.open(request.data["image"])
            im = np.array(ima) / 255
            im = np.expand_dims(im, axis=0)
            pred_bboxes, pred_labels, pred_scores = model.predict(im)
            width, height = ima.size
            bboxes = bbox_utils.denormalize_bboxes(pred_bboxes[0], height, width)
            filename = os.path.split(str(serializer.data["image"]))[-1]
            PredictionAPI.__result_id = int(serializer.data["id"])
            self.__giveBB(filename, ima, bboxes, pred_labels[0], pred_scores[0])
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    def get(self, request, format=None):
        return JsonResponse({
            "id": PredictionAPI.__result_id,
            "image": PredictionAPI.__result_path,
            "objects": PredictionAPI.__result_object
        })
