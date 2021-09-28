#!/usr/bin/python3

# Script to wrap the Mask R-CNN, and to train and run it on Deep Vision data

import os
import pwd

IMAGENAME='maskrcnn'
DATAIN='/home/ketil/src/imagesim-docker'
DATAOUT='/home/ketil/src/maskrcnn/'

USERID=os.getuid()
GROUPID=os.getgid()
USERNAME=pwd.getpwuid(USERID).pw_name
RUNTIME='' # '--gpus device=0'

# print(USERID, GROUPID, USERNAME)

class Model:

    def docker_run(args=''):
        os.system(f'docker run {RUNTIME} --rm --user {USERID}:{GROUPID} -v {DATAIN}:/data:ro -v {DATAOUT}:/project -it {USERNAME}-{IMAGENAME} {args}')

    def docker_build(args):
        os.system(f'docker build --build-arg user={USERNAME} --build-arg uid={USERID} --build-arg gid={GROUPID} -t {USERNAME}-{IMAGENAME} .')

    def train():
        '''Train the network'''
        # if no initial weights: python3 /src/download_weights.py
        docker_run('python3 /src/train.py')

    def check():
        '''Verify that data is in place and that the output doesn't exist'''
        pass

    def predict(target, output):
        '''Run a trained network on the data in target'''
        pass

    def test():
        '''Run tests'''
        docker_run('python3 /src/test.py')
        pass

    def status():
        '''Print the current training status'''
        # check if docker exists
        # check if network is trained (and validation accuracy?)
        # check if data is present for training
        # check if test data is present
        # check if test output is present (and report)
        pass

if __name__ == '__main__':
    import argparse
    import sys
    
    if sys.argv[1] == 'train':
        # p = argparse.ArgumentParser(description='Train Mask R-CNN')
        train()
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'predict':
        predict()
    elif sys.argv[1] == 'check':
        check()
    elif sys.argv[1] == 'status':
        status()
    elif sys.argv[1] == 'build':
        docker_build()
    else:
        error('Usage: {sys.argv[0]} [check,status,train,predict] ...')

