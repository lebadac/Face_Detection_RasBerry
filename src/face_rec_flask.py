from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template , request
from flask_cors import CORS, cross_origin
import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = './Models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

with tf.Graph().as_default():

    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Get input and output tensors
        # images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        # embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        # phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        # embedding_size = embeddings.get_shape()[1]
        # pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")



app = Flask(__name__)
CORS(app)



@app.route('/')
@cross_origin()
def index():
    return "OK!";

@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        name = "Unknown"
        f = request.form.get('image')
        w = int(request.form.get('w'))
        h = int(request.form.get('h'))

        decoded_string = base64.b64decode(f)
        frame = np.frombuffer(decoded_string, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]

        if faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)

            for i in range(faces_found):
                bb[i][0] = int(det[i][0])
                bb[i][1] = int(det[i][1])
                bb[i][2] = int(det[i][2])
                bb[i][3] = int(det[i][3])

                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                if cropped.size == 0:
                    continue

                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                if best_class_probabilities[0] > 0.5:
                    name = class_names[best_class_indices[0]]
                else:
                    name = "Unknown"

                return {
                    "name": name,
                    "box": bb[i].tolist(),
                    "confidence": float(best_class_probabilities[0])  # thêm độ chính xác
                }

        return {
            "name": "Unknown",
            "box": None,
            "confidence": 0.0  # Không có khuôn mặt
        }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='8000')

