import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
from tensorflow.keras.preprocessing import image

warnings.filterwarnings('ignore')

def predict_alzheimer(model_path, image_path):
    model = tf.saved_model.load(model_path)
    print(model.signatures.keys())  # To inspect available signatures
    infer = model.signatures['serving_default']
    print(infer.structured_outputs)  # To see structured outputs and correct output key

    if not image_path:
        print("No image path provided.")
        return
    
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    output_key = list(infer.structured_outputs.keys())[0]  # Assuming the first key is the output
    predictions = infer(tf.constant(img_array))[output_key].numpy()
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    class_labels = {
        0: 'Mild Demented',
        1: 'Moderate Demented',
        2: 'Non Demented',
        3: 'Very Mild Demented'
    }
    predicted_class_name = class_labels[predicted_class_index]

    img_loaded = cv2.imread(image_path)
    img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
    plt.imshow(img_loaded)
    plt.title(f'Predicted Class: {predicted_class_name}')
    plt.axis('off')
    plt.show()

    print(f"The predicted class for the provided image is: {predicted_class_name}")

def main(image_path):
    model_path = 'model/alzheimer_resnet50/content\model/alzheimer_resnet50'
    predict_alzheimer(model_path, image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict Alzheimer\'s disease from MRI image.')
    parser.add_argument('image_path', type=str, help='Path to the MRI image file')
    args = parser.parse_args()

    main(args.image_path)
