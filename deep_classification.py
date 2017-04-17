#! /usr/bin/env python
"""
Loads different types of convolutional neural networks and applies them to a
still image, a video or the webcam.

Usage:
    python deep_classification.py <source> <model> --skip <n_frames>

<source> can be 'webcam', a video file or a JPEG file.
<model> is the name of the network model to be used from 'networks.ini'.
If '--skip' is specified, the specified number of frames is skipped before
applying the network again.

Press ESC or 'q' to exit and 'p' to take a screenshot.
"""
from __future__ import print_function, division

import argparse
import os
import sys
import time

try:
    import ConfigParser as configparser
except ImportError:   # Python 3
    import configparser

import numpy as np
import cv2

CAFFE_ROOT = ''
try:
    CAFFE_ROOT = os.environ['CAFFE_ROOT']
except ImportError:
    print("CAFFE_ROOT env not found. Using default path './caffe'.")

if not os.path.isdir(CAFFE_ROOT):
    print("Directory: {0} not found, unable to use Caffe. Exiting...".
          format(CAFFE_ROOT))
    exit()

sys.path.insert(0, CAFFE_ROOT + '/python')

import caffe

# check if the detection module is available
# Hack: put it always after sys.path to check for Caffe location only once
try:
    import yolo_detection as yolo
    USE_YOLO = True
except ImportError:
    USE_YOLO = False


### Classification ###
######################

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)


class DeepLabeler(object):
    """ given an image it returns a list of tags with associated likelihood """

    def __init__(self, model_file, weights, labels=None, **kwargs):
        self.net = caffe.Net(model_file, weights, caffe.TEST)

        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))

        mode = kwargs.get("mode", "caffe")
        if mode == "yolo":
            self.transformer.set_raw_scale('data', 1.0 / 255.0)
        else:
            mean_pixel = kwargs.get("mean_pixel", None)
            if mean_pixel is not None:
                self.transformer.set_mean('data', mean_pixel)

        self.labels = labels


    def process(self, src):
        """ get the output for the current image """
        input_data = np.asarray([self.transformer.preprocess('data', src)])
        net_outputs = self.net.forward_all(data=input_data)
        net_output = net_outputs[net_outputs.keys()[0]]   # get first out layer

        if len(net_output.shape) > 2:
            net_output = np.squeeze(net_output)[np.newaxis, :]

        ids = np.argsort(net_output[0])[-1:-6:-1]
        predictions = [(self.labels[cls_id], net_output[0][cls_id])
                       for cls_id in ids]

        print('predicted classes:', predictions)

        return predictions


    @staticmethod
    def draw_predictions(image, predictions):
        """ draw the name and scores for the top 5 predictions """

        roi = image[10:150, 10:500]
        rect = np.zeros((140, 490, 3), dtype=np.uint8)
        alpha = 0.7

        cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0.0, roi)

        for (i, (cls_id, value)) in enumerate(predictions):
            cv2.rectangle(image, (15, 15 + 26*i),
                          (15 + int(175*value), 35 + 26*i), COLOR_GREEN, -1)
            cv2.putText(image, cls_id, (190, 35 + 26*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE)

        return image


### model selection ###
#######################


def load_labels(label_file):
    """ load list of labels from file (one per line) """
    labels = []
    with open(label_file, 'r') as handle:
        labels = [line.strip() for line in handle]
    return labels


def load_network(config_filename, model_name):
    """ load network parameters and create processor instance """
    base_path = os.path.dirname(config_filename)

    config = configparser.ConfigParser()
    config.read(config_filename)

    if model_name not in config.sections():
        raise ValueError(
            "Model {0} not available in {1}".format(model_name, config_filename))

    section = dict(config.items(model_name))
    model_type = section['type']
    model_file = os.path.join(base_path, os.path.normpath(section['model']))
    weights = os.path.join(base_path, os.path.normpath(section['weights']))
    label_file = os.path.join(base_path, os.path.normpath(section['labels']))

    labels = load_labels(label_file)
    mean_pixel = (np.array([int(ch) for ch in section['mean'].split(',')])
                  if 'mean' in section.keys() else None)

    if section.get('device', "gpu") == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if   model_type == "detect_yolo":
        return yolo.YoloDetector(model_file, weights, labels)
    elif model_type == "class":
        return DeepLabeler(model_file, weights, labels, mean_pixel=mean_pixel)
    elif model_type == "class_yolo":
        return DeepLabeler(model_file, weights, labels, mode="yolo")
    else:
        raise ValueError("Unrecognized type {0} for network {1}".format(model_type, model_name))


### Demo UI ###
###############

def draw_fps(image, fps):
    """ Draw the running average of the frame rate for the last predictions """

    roi = image[10:45, 530:730]
    rect = np.zeros((35, 200, 3), dtype=np.uint8)
    alpha = 0.7

    cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0.0, roi)
    cv2.putText(image, "FPS: %.3f" % fps, (530, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    return image


def aspect_ratio(image):
    """ compute the aspect ratio from the image size """
    width, height = image.shape[:2]
    return height / width


# Constants #
#############

KEY_ESC = 27           # key code for ESC
MAX_IMAGE_SIDE = 640   # max height or width allowed for the image


def main_loop(processor, source_str, frame_skip=1):
    """ applies the model to all the images from the source """

    video_capture = cv2.VideoCapture(0 if source_str == 'webcam' else source_str)

    fps = 0.0
    delay = 30

    i = 0
    while True:
        ret, image = video_capture.read()

        if not ret and i == 0:
            raise IOError("source {} not found".format(source_str))

        if ret:
            frame = cv2.resize(
                image, (int(MAX_IMAGE_SIDE * aspect_ratio(image)), MAX_IMAGE_SIDE))

            if i % frame_skip == 0:
                t_start = time.time()
                predictions = processor.process(frame)
                fps = 0.5 * fps + 0.5 / (time.time() - t_start)

            frame = processor.draw_predictions(frame, predictions)
            frame = draw_fps(frame, fps)
            i += 1
        else:
            delay = -1

        cv2.imshow('Video', frame)
        keypress = cv2.waitKey(delay) & 0xFF

        if   keypress == ord('p'):              # screenshot
            source_name = os.path.basename(os.path.splitext(source_str)[0])
            cv2.imwrite("{0}.{1:03d}.png".format(source_name, i), frame)
        elif keypress in [ord('q'), KEY_ESC]:   # exit
            cv2.destroyAllWindows()
            return


def get_script_path():
    """ returns the directory of the current script """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def main():
    """ entry point function """
    parser = argparse.ArgumentParser(
        description='Deep Classification Demo.',
        epilog="based on Caffe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', type=str, default='webcam', help='video source file')
    parser.add_argument('net', type=str, default='caffenet',
                        help='pretrained network to use (from network.ini)')
    parser.add_argument('--skip', type=int, default=1, help='skip every n frames')
    args = parser.parse_args()

    configuration_file = os.path.join(get_script_path(), "networks.ini")
    main_loop(load_network(configuration_file, args.net), args.source, args.skip)


if __name__ == '__main__':
    main()
