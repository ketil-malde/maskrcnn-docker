Dockerized version of the Mattermost implementation of Mask R-CNN

Should only depend on having a working docker environment (and user
rights to run docker) an GPU drivers (CUDA) installed in the host
system.

The ´maskrcnn´ script contains everything needed for building and running, specifically:

    maskrcnn build - builds the docker instance for subsequent commands
    maskrcnn train - run a set of training epochs
    maskrcnn test  - test the latest trained network

Most interesting configurations should be in ´maskrcnn´ and in ´src/config.py´.

The default is configured to learn from the output of imagesim-docker.
