#!/bin/sh
#
# Downloads the weights for the networks supported by the demo if they are not
# available.
#
# Usage:
#  ./download_models.sh [all|caffenet|googlenet|googlenet_places205|squeezenet|
#                        yolo_tiny|yolo]
#

load_caffe_model () {
    MODEL_LABEL="$1"
    MODEL_NAME="$2"
    SHA1="$3"
    MODEL_PATH=${MODEL_NAME}/${MODEL_NAME}.caffemodel
    if ! [ -f ${MODEL_PATH} ]; then
        echo "\nDownloading ${MODEL_LABEL}..."
        wget http://dl.caffe.berkeleyvision.org/${MODEL_NAME}.caffemodel -P ${MODEL_NAME}
        echo "CHECKSUM"
        echo "${SHA1} *${MODEL_NAME}/${MODEL_NAME}.caffemodel" \
            | sha1sum -c -
    fi
}


if [ "$1" = "all" ]; then
    MODELS="caffenet googlenet googlenet_places205 squeezenet yolo_tiny yolo"
elif [ $# -eq 0 ]; then
    MODELS=""
else
    MODELS="$@"
fi

for model in $MODELS
do
    case $model in
        "caffenet")
           load_caffe_model "CaffeNet" "bvlc_reference_caffenet" "4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46"
        ;;
        "googlenet")
            load_caffe_model "GoogleNet" "bvlc_googlenet" "405fc5acd08a3bb12de8ee5e23a96bec22f08204"
        ;;
        "googlenet_places205")
            MODEL_NAME="googlenet_places205"
            MODEL_PATH=${MODEL_NAME}/googlelet_places205_train_iter_2400000.caffemodel
            if ! [ -f ${MODEL_PATH} ]; then
                echo "\nDownloading GoogleNet Places205..."
                wget http://places.csail.mit.edu/model/googlenet_places205.tar.gz -P ${MODEL_NAME}
                tar -zxvf ${MODEL_NAME}/googlenet_places205.tar.gz ${MODEL_PATH} -C ${MODEL_NAME}
            fi
        ;;
        "squeezenet")
            if ! [ -d SqueezeNet ]; then
                echo "\nCloning SqueezeNet repository..."
                git clone https://github.com/DeepScale/SqueezeNet.git
            fi
        ;;
        "yolo_tiny")
            if ! [ -f yolo/yolo_tiny.caffemodel ]; then
                echo "\nDownloading YOLO tiny..."
                wget https://drive.google.com/file/d/0Bzy9LxvTYIgKNFEzOEdaZ3U0Nms/view?usp=sharing -P yolo
            fi
        ;;
        "yolo")
            if ! [ -f yolo/yolo.caffemodel ]; then
                echo "\nDownloading YOLO tiny..."
                wget https://drive.google.com/file/d/0Bzy9LxvTYIgKMXdqS29HWGNLdGM/view?usp=sharing -P yolo
            fi
        ;;
    esac
done
