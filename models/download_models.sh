#!/bin/sh
#
# Downloads the weights for the networks supported by the demo if they are not
# available.
#
# Usage:
#  ./download_models.sh [all|caffenet|googlenet|googlenet_places205|squeezenet]
#

# fallback to wget if curl not available (e.g. default Linux installations) 
CURL='curl' 
if ! [ -x "$(command -v curl)" ]; then 
    CURL='wget' 
fi 

load_caffe_model () {
    MODEL_LABEL="$1"
    MODEL_NAME="$2"
    SHA1="$3"
    MODEL_PATH=${MODEL_NAME}/${MODEL_NAME}.caffemodel
    if ! [ -f ${MODEL_PATH} ]; then
        printf "\nDownloading ${MODEL_LABEL}...\n"
        ${CURL} http://dl.caffe.berkeleyvision.org/${MODEL_NAME}.caffemodel \
            -o ${MODEL_PATH}
        printf "CHECKSUM\n"
        echo "${SHA1} *${MODEL_NAME}/${MODEL_NAME}.caffemodel" \
            | sha1sum -c -
    fi
}


if [ "$1" = "all" ]; then
    MODELS="caffenet googlenet googlenet_places205 squeezenet"
elif [ $# -eq 0 ]; then
    MODELS=""
else
    MODELS="$@"
fi

for model in $MODELS
do
    case $model in
        "caffenet")
            load_caffe_model "CaffeNet" "bvlc_reference_caffenet" \
                "4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46"
        ;;
        "googlenet")
            load_caffe_model "GoogleNet" "bvlc_googlenet" \
                "405fc5acd08a3bb12de8ee5e23a96bec22f08204"
        ;;
        "googlenet_places205")
            MODEL_NAME="googlenet_places205"
            MODEL_PATH=${MODEL_NAME}/googlelet_places205_train_iter_2400000.caffemodel
            if ! [ -f ${MODEL_PATH} ]; then
                printf "\nDownloading GoogleNet Places205...\n"
                ${CURL} http://places.csail.mit.edu/model/googlenet_places205.tar.gz \
                    -o googlenet_places205.tar.gz
                tar -zxvf googlenet_places205.tar.gz ${MODEL_PATH} \
                    -C ${MODEL_NAME}
            fi
        ;;
        "squeezenet")
            if ! [ -d SqueezeNet ]; then
                printf "\nCloning SqueezeNet repository...\n"
                git clone https://github.com/DeepScale/SqueezeNet.git
            fi
        ;;
    esac
done
