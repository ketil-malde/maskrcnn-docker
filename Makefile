# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = maskrcnn
CONFIG    = tensorflow
COMMAND   = bash
DISKS     = -v $(PWD)/../imagesim-docker:/data:ro -v $(PWD):/project
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)
PORT      = -p 8888:8888
RUNTIME   = --gpus=1
# --runtime=nvidia 
# No need to change anything below this line

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

.PHONY: .docker test train

.docker: docker/Dockerfile-$(CONFIG)
	docker build $(USERCONFIG) -t $(USERNAME)-$(IMAGENAME) -f docker/Dockerfile-$(CONFIG) docker

WEIGHTS = mask_rcnn_coco.h5

# Using -it for interactive use
RUNCMD=docker run $(RUNTIME) --rm --user $(USERID):$(GROUPID) $(PORT) $(SSHFSOPTIONS) $(DISKS) -it $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) $(COMMAND)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip '$(hostname -I)' --port 8888

$(WEIGHTS): src/download_weights.py
	$(RUNCMD) python3 src/download_weights.py

train: .docker src/test.py $(WEIGHTS)
	$(RUNCMD) python3 src/train.py

test: .docker src/test.py $(WEIGHTS)
	$(RUNCMD) python3 src/test.py

