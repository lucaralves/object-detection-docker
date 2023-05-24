import os
import sys
import cv2
import tensorflow as tf

# Project structure.
PROJECT_DIR = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_FILE_DIR = os.path.join(MODEL_DIR, "saved_model.pb")
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")

def get_input_shape(graph, input_tensor_name):
    for input_tensor in graph.inputs:
        if input_tensor.name.split(':')[0] == input_tensor_name:
            input_tensor_shape = input_tensor.shape.as_list()
            return input_tensor_shape
    return None

if __name__ == "__main__":

    # Import the TF graph
    loaded_model = tf.saved_model.load(MODEL_DIR)

    # Get input and output info of the graph.
    graph = loaded_model.signatures["serving_default"]
    input_tensor_names = [input_tensor.name.split(':')[0] for input_tensor in graph.inputs]
    output_tensor_names = [output_tensor.name.split(':')[0] for output_tensor in graph.outputs]

    # Print input tensor names and shapes
    print("Input Tensor Names and Shapes:")
    for input_tensor_name in input_tensor_names:
        input_tensor_shape = get_input_shape(graph, input_tensor_name)
        print(f"{input_tensor_name}: {input_tensor_shape}")

    # Carrega-se a imagem.
    image_path = os.path.join(IMAGES_DIR, "eee.jpg")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    print(image.shape.as_list())

    # Pre-process the image.
    resized_image = tf.image.resize(image, [224, 224])
    expanded_image = tf.expand_dims(resized_image, 0)
    print(expanded_image.shape.as_list())

    # Encode the processed image to a string tensor.
    encoded_image = tf.io.encode_jpeg(tf.cast(expanded_image[0], tf.uint8))
    encoded_image = encoded_image[tf.newaxis]

    # Make the prediction.
    outputs = graph(image_bytes=encoded_image, key=tf.expand_dims(tf.convert_to_tensor("myKey"), 0))

    # Process the output of the model.
    detection_boxes = outputs['detection_boxes'].numpy()
    detection_classes_as_text = outputs['detection_classes_as_text'].numpy()
    detection_scores = outputs['detection_scores'].numpy()
    index_of_detections = []
    i = 0
    for detection_score in detection_scores[0]:
        if detection_score >= 0.8:
            index_of_detections.append(i)
        i = i + 1

    # Process the image.
    img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
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

    # Show the image on UI.
    while True:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        sys.exit()
    cv2.destroyAllWindows()