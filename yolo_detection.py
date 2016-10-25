"""
Loads a YOLO detector network converted in Caffe and visualizes the detected
objects.
Code adapted from:

   https://github.com/xingwangsfu/caffe-yolo

Moved in a separate module for licensing reasons.
"""
from __future__ import print_function, division

import numpy as np
import cv2

import matplotlib.cm as cmx
import matplotlib.colors as colors

import caffe


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


def parse_yolo_output(output, img_size, num_classes):
    """ convert the output of YOLO's last layer to boxes and confidence in each
    class """

    num_boxes = 2
    grid_size = 7

    sc_offset = grid_size * grid_size * num_classes
    box_offset = sc_offset + grid_size * grid_size * num_boxes

    class_probs = np.reshape(output[0:sc_offset], (grid_size, grid_size, num_classes))
    confidences = np.reshape(output[sc_offset:box_offset], (grid_size, grid_size, num_boxes))

    probs = np.zeros((grid_size, grid_size, num_boxes, num_classes))
    for i in range(num_boxes):
        for j in range(num_classes):
            probs[:, :, i, j] = class_probs[:, :, j] * confidences[:, :, i]

    boxes = get_boxes(output[box_offset:], img_size, grid_size, num_boxes)

    return boxes, probs


def get_candidate_objects(output, img_size, classes):
    """ convert network output to bounding box predictions """

    threshold = 0.2
    iou_threshold = 0.5

    boxes, probs = parse_yolo_output(output, img_size, len(classes))

    filter_mat_probs = (probs >= threshold)
    filter_mat_boxes = np.nonzero(filter_mat_probs)[0:3]
    boxes_filtered = boxes[filter_mat_boxes]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes]

    idx = np.argsort(probs_filtered)[::-1]
    boxes_filtered = boxes_filtered[idx]
    probs_filtered = probs_filtered[idx]
    classes_num_filtered = classes_num_filtered[idx]

    # Non-Maxima Suppression: greedily suppress low-scoring overlapped boxes
    for i, box_filtered in enumerate(boxes_filtered):
        if probs_filtered[i] == 0:
            continue
        for j in range(i+1, len(boxes_filtered)):
            if iou(box_filtered, boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = (probs_filtered > 0.0)
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = [[classes[class_id], box[0], box[1], box[2], box[3], prob]
              for class_id, box, prob in
              zip(classes_num_filtered, boxes_filtered, probs_filtered)]

    return result


def iou(box1, box2):
    """ compute intersection over union score """
    int_tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
             max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    int_lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
             max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    intersection = max(0.0, int_tb) * max(0.0, int_lr)

    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


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

    def __init__(self, model_file, weights, labels):
        self.net = caffe.Net(model_file, weights, caffe.TEST)

        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 1.0 / 255.0)

        self.classes = labels
        self.colormap = get_colormap(len(self.classes))


    def process(self, src):
        """ get the output for the current image """
        input_data = np.asarray([self.transformer.preprocess('data', src)])
        out = self.net.forward_all(data=input_data)

        return get_candidate_objects(out['result'][0], src.shape, self.classes)


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

            print(" class : {}, [x,y,w,h]=[{:d},{:d},{:d},{:d}], Confidence = {}".\
                  format(class_id, box_x, box_y, box_w, box_h, str(result[5])))
            box = (max(box_x-box_w//2, 0), max(box_y-box_h//2, 0),
                   min(box_x+box_w//2, img_width), min(box_y+box_h//2, img_height))

            self.draw_box_(image, class_id, box, result[5])

        return image


if __name__ == '__main__':
    pass
