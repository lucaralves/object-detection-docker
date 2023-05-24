import os
import cv2
import base64
from fastapi import FastAPI, UploadFile
import numpy as np
import tensorflow as tf

# Project structure.
PROJECT_DIR = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_FILE_DIR = os.path.join(MODEL_DIR, "saved_model.pb")

# Import the TF graph.
loaded_model = tf.saved_model.load(MODEL_DIR)

# Web app entrypoint.
app = FastAPI()

# Global variables.
index_of_detections = []
detection_boxes = None
detection_classes_as_text = None
detection_scores = None

# Get the input shape of the model.
def get_input_shape(graph, input_tensor_name):

    for input_tensor in graph.inputs:
        if input_tensor.name.split(':')[0] == input_tensor_name:
            input_tensor_shape = input_tensor.shape.as_list()
            return input_tensor_shape
    return None

# Convert the output of the model in numpy arrays and get the index of the most confident detections.
def processOutputOfModel(outputs):

    global detection_boxes
    global detection_classes_as_text
    global detection_scores

    detection_boxes = outputs['detection_boxes'].numpy()
    detection_classes_as_text = outputs['detection_classes_as_text'].numpy()
    detection_scores = outputs['detection_scores'].numpy()
    i = 0
    for detection_score in detection_scores[0]:
        if detection_score >= 0.8:
            index_of_detections.append(i)
        i = i + 1

# Draw the bounding boxes.
def processImage(img):

    global detection_boxes
    global detection_classes_as_text

    color = (0, 255, 0)
    thickness = 1
    for index_of_detection in index_of_detections:
        start_point = (int(detection_boxes[0][index_of_detection][1] * img.shape[1]),
                       int(detection_boxes[0][index_of_detection][0] * img.shape[0]))
        end_point = (int(detection_boxes[0][index_of_detection][3] * img.shape[1]),
                     int(detection_boxes[0][index_of_detection][2] * img.shape[0]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.putText(img, str(detection_classes_as_text[0][index_of_detection]),
                    (int(detection_boxes[0][index_of_detection][1] * img.shape[1]),
                     int(detection_boxes[0][index_of_detection][0] * img.shape[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 36, 255), 2)

    return img

@app.post("/postImage")
async def postController(uploadFile: UploadFile):

    # Get input and output info of the graph.
    graph = loaded_model.signatures["serving_default"]
    input_tensor_names = [input_tensor.name.split(':')[0] for input_tensor in graph.inputs]

    # Print input tensor names and shapes.
    print("Input Tensor Names and Shapes:")
    for input_tensor_name in input_tensor_names:
        input_tensor_shape = get_input_shape(graph, input_tensor_name)
        print(f"{input_tensor_name}: {input_tensor_shape}")

    # Carrega-se a imagem.
    image = await uploadFile.read()
    tensorImage = tf.image.decode_jpeg(image, channels=3)
    print(tensorImage.shape.as_list())

    # Pre-process the image.
    resized_image = tf.image.resize(tensorImage, [224, 224])
    expanded_image = tf.expand_dims(resized_image, 0)
    print(expanded_image.shape.as_list())

    # Encode the processed image to a string tensor.
    encoded_image = tf.io.encode_jpeg(tf.cast(expanded_image[0], tf.uint8))
    encoded_image = encoded_image[tf.newaxis]

    # Make the prediction.
    outputs = graph(image_bytes=encoded_image, key=tf.expand_dims(tf.convert_to_tensor("myKey"), 0))
    processOutputOfModel(outputs)

    # Draw the bounding boxes on the image.
    outputImage = processImage(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR))

    # Encode the image to base64.
    _, encoded_image = cv2.imencode(".jpg", outputImage)
    base64Image = base64.b64encode(encoded_image.tobytes())

    return {"base64": base64Image}