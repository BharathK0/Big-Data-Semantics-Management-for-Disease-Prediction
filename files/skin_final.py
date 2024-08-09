import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.preprocessing import image

warnings.filterwarnings('ignore')

def predict_skin_disease(model_path, image_path):
    """Load model, prepare image, and make predictions on the provided image path."""
    try:
        # Load the TensorFlow SavedModel for inference
        loaded = tf.saved_model.load(model_path)
        infer = loaded.signatures['serving_default']
        print("Model loaded successfully from", model_path)
        print("Available output keys:", infer.structured_outputs)

        if not image_path:
            print("No image path provided.")
            return

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(28, 28))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Predict the class of a skin lesion using the loaded model
        output_key = list(infer.structured_outputs.keys())[0]  # Dynamically get the output key
        predictions = infer(tf.constant(img_array))[output_key].numpy()
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Define class labels
        class_labels = {
            0: 'Actinic keratosis',
            1: 'Basal cell carcinoma',
            2: 'Benign keratosis-like lesions',
            3: 'Dermatofibroma',
            4: 'Melanocytic nevi',
            5: 'Melanoma',
            6: 'Vascular lesion'
        }
        predicted_class_name = class_labels[predicted_class_index]

        # Display the image and prediction
        img_loaded = cv2.imread(image_path)
        img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
        plt.imshow(img_loaded)
        plt.title(f'Predicted Class: {predicted_class_name}')
        plt.axis('off')
        plt.show()

        print(f"The predicted class for the provided image is: {predicted_class_name}")

    except Exception as e:
        print("An error occurred:", str(e))

def main(image_path):
    model_path = 'model/skin_disease_model/content/model/skin_disease_model'
    predict_skin_disease(model_path, image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict skin disease from an image.')
    parser.add_argument('image_path', type=str, help='Path to the skin image file')
    args = parser.parse_args()

    main(args.image_path)
