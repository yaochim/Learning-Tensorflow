import os
import urllib.request
import tarfile
import cv2
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into GraphDef

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

# videofilename = "6821175666942859224.mp4"
videofilename = 2
labels_dir = f'{os.getcwd()}/models/research/object_detection/'

# source_video = cv2.VideoCapture(videofilename)
# source_video = cv2.VideoCapture(2)
#
# while source_video.isOpened():
#     ret, frame = source_video.read()
#
#     cv2.imshow("inter", frame)
#
#     if cv2.waitKey(40) == 27:
#         break
#
# cv2.destroyAllWindows()
# source_video.release()


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(labels_dir + 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def setup():
    # download Tensorflow model tar package from  Tensorflow website
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    # open the tar and extract the inference graph
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


def detect(videofilename):

    # instanctiate a TensorFlow graph
    detection_graph = tf.Graph()
    # set the graph as default graph
    with detection_graph.as_default():
        od_graph_def = utils_ops.tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with utils_ops.tf.Session(graph=detection_graph) as sess:

            cap = cv2.VideoCapture(videofilename)

            while True:
                ret, color_frame = cap.read()

                # color_frame = cv2.resize(frame, (640, 360))
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # Define input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('object detection', color_frame)
                # output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                # out.write(output_rgb)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print(f'quit')
                    break

            cv2.destroyAllWindows()
            cap.release()


def main():
    setup()
    detect(videofilename)


if __name__ == '__main__':
    main()
