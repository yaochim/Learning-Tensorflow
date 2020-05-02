# Tensorflow

## Dependencies
Additional python modules required, which I have put in to the requirements.txt
- tensorflow
- matplotlib
- opencv-python
- pycocotools
- cython


## Setup steps
Recommend to use pipenv

1. Set environment and install python module dependencies from above
```sh
$ pipenv --python 3.7
$ pipenv install -r requirements.txt
```

2. Download TensorFlow models repository.
```sh
git clone https://github.com/tensorflow/models.git
```

3. Install profobuf on the main machine OS (not in the Python virtualenv).
protobuf from pip does not contain the protoc command needed so manual install is needed
```sh
cd models/research
wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
unzip protobuf.zip
```

4. Compile the models using the protoc command downloaded from above step.
```sh
./bin/protoc object_detection/protos/*.proto --python_out=.
```


## Code overview

1. Import necessary libraries
2. Set global vars for Tensorflow models etc.
3. Function to extract Tensorflow models
4. Function to detect (inference)
5. Function to process video or stream or image


## Gotcha's

The python module installed is tensorflow v2.1.x , however the models repository is still set to use tensorflow 1.x, there for some tweaks/patches are require to make the code backward compatible, see lines 14-17.
Where "tf" was used it was replaced with utils_ops in order to use v1 code.



## Resources

#### Tensorflow references:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
https://www.tensorflow.org/api_docs/python/tf
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

#### Tensorflow Pre-trained Models
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
https://github.com/tensorflow/models/archive/master.zip

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz


#### Other references:
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Pet_detector.py
https://www.edureka.co/blog/tensorflow-object-detection-tutorial/
https://towardsdatascience.com/detecting-pikachu-in-videos-using-tensorflow-object-detection-cd872ac42c1d
