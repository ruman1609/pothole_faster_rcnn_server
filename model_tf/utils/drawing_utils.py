import matplotlib.pyplot as plt, tensorflow as tf, random
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from . import bbox_utils

def draw_grid_map(img, grid_map, stride):
    """Drawing grid intersection on given image.
    inputs:
        img = (height, width, channels)
        grid_map = (output_height * output_width, [y_index, x_index, y_index, x_index])
            tiled x, y coordinates
        stride = number of stride
    outputs:
        array = (height, width, channels)
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()

def draw_bboxes_with_labels(img, bboxes, label_indices, probs, labels, font):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in denormalized form
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    colors = []
    for i in range(len(labels)):
        colors.append([random.randint(0, 128), 255, random.randint(0, 128), 255])
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
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
        draw.text((x1 + 4, y1 + 2), label_text, fill=color, font=font)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_predictions(dataset, pred_bboxes, pred_labels, pred_scores, labels, batch_size):
    font = font_manager.FontProperties(family='sans-serif', weight='bold')
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 12)
    for batch_id, image_data in enumerate(dataset):
        imgs, _, _ = image_data
        img_size = imgs.shape[1]
        start = batch_id * batch_size
        end = start + batch_size
        batch_bboxes, batch_labels, batch_scores = pred_bboxes[start:end], pred_labels[start:end], pred_scores[start:end]
        for i, img in enumerate(imgs):
            denormalized_bboxes = bbox_utils.denormalize_bboxes(batch_bboxes[i], img_size, img_size)
            draw_bboxes_with_labels(img, denormalized_bboxes, batch_labels[i], batch_scores[i], labels, font)
