"""
Loads a YOLO detector network converted in Caffe and visualizes the detected
objects.
Code adapted from:

   https://github.com/xingwangsfu/caffe-yolo

Moved in a separate module for licensing reasons.
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.cm as cmx
from matplotlib import colors

import cv2

try:
    import caffe
except ImportError:
    print("Not a standalone module; launch 'deep_classification.py' instead.")


def get_boxes(output, img_size, grid_size, num_boxes):
    """ extract bounding boxes from the last layer """

    w_img, h_img = img_size[1], img_size[0]
    boxes = np.reshape(output, (grid_size, grid_size, num_boxes, 4))

    offset = np.tile(np.arange(grid_size)[:, np.newaxis],
                     (grid_size, 1, num_boxes))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] /= 7.0
    # the predicted size is the square root of the box size
    boxes[:, :, :, 2:4] *= boxes[:, :, :, 2:4]

    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img

    return boxes


def parse_yolo_output_v1(output, img_size, num_classes):
    """ convert the output of YOLO's last layer to boxes and confidence in each
    class """

    n_coord_box = 4    # number of coordinates in each bounding box
    grid_size = 7

    sc_offset = grid_size * grid_size * num_classes

    # autodetect num_boxes
    num_boxes = int((output.shape[0] - sc_offset) /
                    (grid_size*grid_size*(n_coord_box+1)))
    box_offset = sc_offset + grid_size * grid_size * num_boxes

    class_probs = np.reshape(output[0:sc_offset],
                             (grid_size, grid_size, num_classes))
    confidences = np.reshape(output[sc_offset:box_offset],
                             (grid_size, grid_size, num_boxes))

    probs = np.zeros((grid_size, grid_size, num_boxes, num_classes))
    for i in range(num_boxes):
        for j in range(num_classes):
            probs[:, :, i, j] = class_probs[:, :, j] * confidences[:, :, i]

    boxes = get_boxes(output[box_offset:], img_size, grid_size, num_boxes)

    return boxes, probs


def logistic(val):
    """ compute the logistic activation """
    return 1.0 / (1.0 + np.exp(-val))


def softmax(val, axis=-1):
    """ compute the softmax of the given tensor, normalizing on axis """
    exp = np.exp(val - np.amax(val, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


def get_boxes_v2(output, img_size, biases):
    """ extract bounding boxes from the last layer (Darknet v2) """
    bias_w, bias_h = biases

    w_img, h_img = img_size[1], img_size[0]
    grid_w, grid_h, num_boxes = output.shape[:3]

    # tweak: add a 0.5 offset to improve localization accuracy
    offset_x = \
        np.tile(np.arange(grid_w)[:, np.newaxis], (grid_h, 1, num_boxes)) - 0.5
    offset_y = np.transpose(offset_x, (1, 0, 2))

    boxes = output.copy()
    boxes[:, :, :, 0] = (offset_x + logistic(boxes[:, :, :, 0])) / grid_w
    boxes[:, :, :, 1] = (offset_y + logistic(boxes[:, :, :, 1])) / grid_h
    boxes[:, :, :, 2] = np.exp(boxes[:, :, :, 2]) * bias_w / grid_w
    boxes[:, :, :, 3] = np.exp(boxes[:, :, :, 3]) * bias_h / grid_h

    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img

    return boxes


def parse_yolo_output_v2(output, img_size, num_classes, anchors):
    """ convert the output of the last convolutional layer (Darknet v2) """
    n_coord_box = 4
    biases = [float(x) for x in anchors.split(',')]
    biases = [biases[::2], biases[1::2]]

    # for each box: coordinates, probs scale, class probs
    num_boxes = output.shape[0] // (n_coord_box + 1 + num_classes)
    output = output.reshape((num_boxes, -1, output.shape[1], output.shape[2]))
    output = output.transpose((2, 3, 0, 1))

    probs = logistic(output[:, :, :, 4:5]) * softmax(
        output[:, :, :, 5:], axis=3)
    boxes = get_boxes_v2(output[:, :, :, :4], img_size, biases)

    return boxes, probs


def parse_yolo_output(output, img_size, num_classes, anchors):
    """ convert the output of YOLO's last layer to boxes and confidence in each
    class """
    if anchors is not None and len(output.shape) == 3:
        return parse_yolo_output_v2(output, img_size, num_classes, anchors)
    if len(output.shape) == 1:
        return parse_yolo_output_v1(output, img_size, num_classes)

    raise ValueError(" output format not recognized")


def get_candidate_objects(output, img_size, classes, anchors=None):
    """ convert network output to bounding box predictions """
    threshold = 0.2
    iou_threshold = 0.4

    boxes, probs = parse_yolo_output(output, img_size, len(classes), anchors)

    filter_mat_probs = (probs >= threshold)
    filter_mat_boxes = np.nonzero(filter_mat_probs)[0:3]
    boxes_filtered = boxes[filter_mat_boxes]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes]

    idx = np.argsort(probs_filtered)[::-1]
    boxes_filtered = boxes_filtered[idx]
    probs_filtered = probs_filtered[idx]
    classes_num_filtered = classes_num_filtered[idx]

    # too many detections - exit
    if len(boxes_filtered) > 1e3:
        print("Too many detections, maybe an error? : {}".format(
            len(boxes_filtered)))
        return []

    probs_filtered = non_maxima_suppression(
        boxes_filtered, probs_filtered, classes_num_filtered, iou_threshold)

    filter_iou = (probs_filtered > 0.0)
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    return [[classes[class_id], box[0], box[1], box[2], box[3], prob]
            for class_id, box, prob in
            zip(classes_num_filtered, boxes_filtered, probs_filtered)]


def non_maxima_suppression(boxes, probs, classes_num, thr=0.2):
    """ greedily suppress low-scoring overlapped boxes """
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        for j in range(i+1, len(boxes)):
            if classes_num[i] == classes_num[j] and iou(box, boxes[j]) > thr:
                probs[j] = 0.0

    return probs


def iou(box1, box2, denom="min"):
    """ compute intersection over union score """
    int_tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    int_lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

    intersection = max(0.0, int_tb) * max(0.0, int_lr)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]

    if denom == "min":
        control_area = min(area1, area2)
    else:
        control_area = area1 + area2 - intersection

    return intersection / control_area


def get_colormap(n_colors):
    """ Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color extracted by the HSV colormap """

    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb(index):
        """ closure mapping index to color - range [0 255] for OpenCV """
        return 255.0 * np.array(scalar_map.to_rgba(index))

    return map_index_to_rgb


class YoloDetector(object):
    """ given an image it returns a list detected objects with class and
    likelihood. Implemented using the YOLO network.
    Adapted from https://github.com/xingwangsfu/caffe-yolo """

    def __init__(self, model_file, weights, labels, anchors=None):
        self.net = caffe.Net(model_file, weights, caffe.TEST)
        self.anchors = anchors

        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 1.0 / 255.0)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        self.classes = labels
        self.colormap = get_colormap(len(self.classes))

    def process(self, src):
        """ get the output for the current image """
        input_data = np.asarray([self.transformer.preprocess('data', src)])
        out = self.net.forward_all(data=input_data)

        return get_candidate_objects(out['result'][0], src.shape, self.classes,
                                     self.anchors)

    def draw_box_(self, img, name, box, score):
        """ draw a single bounding box on the image """
        xmin, ymin, xmax, ymax = box

        color = self.colormap(self.classes.index(name))
        box_tag = '{} : {:.2f}'.format(name, score)
        text_x, text_y = 5, 7

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (xmin, ymin-boxsize[1]-text_y),
                      (xmin+boxsize[0]+text_x, ymin), color, -1)
        cv2.putText(img, box_tag, (xmin+text_x, ymin-text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_predictions(self, image, predictions):
        """ draw bounding boxes in the form (class, box, score) """
        img_width, img_height = image.shape[1], image.shape[0]

        print("Detections:")
        for result in predictions:
            class_id = result[0]
            box_x, box_y, box_w, box_h = [int(v) for v in result[1:5]]

            print(" class : {}, [x,y,w,h]=[{:d},{:d},{:d},{:d}], "
                  "Confidence = {}".format(
                      class_id, box_x, box_y, box_w, box_h, str(result[5])))
            box = (max(box_x-box_w//2, 0), max(box_y-box_h//2, 0),
                   min(box_x+box_w//2, img_width),
                   min(box_y+box_h//2, img_height))

            self.draw_box_(image, class_id, box, result[5])

        return image


if __name__ == '__main__':
    pass
