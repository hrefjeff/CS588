#!/bin/bash

REGISTRY_SERVER="docker.io"
IMAGE_NAME="jeffwayne256/python-base-img"
IMAGE_TAG="latest"

BIND_MNT_CMD="-v $PWD:/usr/src -w /usr/src"

CMD="/usr/local/bin/docker run -it --rm ${BIND_MNT_CMD} $REGISTRY_SERVER/$IMAGE_NAME:$IMAGE_TAG"

eval $CMD
