# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = maskrcnn
COMMAND   = bash
DISKS     = -v $(PWD)/../imagesim-docker:/data:ro -v $(PWD):/project
PORT      =
# For jupyter:
# PORT    = -p 8888:8888
NETWORK =
# Sometimes necessary for networking to work:
# NETWORK   = --network host
GPU       = 0
# RUNTIME   =
RUNTIME   = --gpus device=$(GPU)
# No need to change anything below this line
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

.PHONY: .docker test train

.docker: docker/Dockerfile
	docker build $(USERCONFIG) $(NETWORK) -t $(USERNAME)-$(IMAGENAME) docker

WEIGHTS = mask_rcnn_coco.h5

# Using -it for interactive use
RUNCMD=docker run $(RUNTIME) $(NETWORK) --rm --user $(USERID):$(GROUPID) $(PORT) $(SSHFSOPTIONS) $(DISKS) -it $(USERNAME)-$(IMAGENAME)

# Starts and interactive shell by default (COMMAND = bash)
default: .docker
	$(RUNCMD) $(COMMAND)

$(WEIGHTS): .docker
	$(RUNCMD) python3 /src/download_weights.py

train: .docker $(WEIGHTS)
	$(RUNCMD) python3 /src/train.py

test: .docker $(WEIGHTS)
	$(RUNCMD) python3 /src/test.py
