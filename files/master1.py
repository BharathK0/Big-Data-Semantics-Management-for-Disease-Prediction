import argparse
import tensorflow as tf
import time
import malaria
import predict_alzh
import skin_final
import lung
from finaldiabetes import Diabetes
from text_recognizer import text_reco

# Load the SavedModel once
model = tf.saved_model.load('savedmodel1')

# Define labels based on provided labels.txt
labels = {
    0: 'skin',
    1: 'document',
    2: 'lung',
    3: 'malaria',
    4: 'alziemer'
}

# Preprocess the image once
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Classify the image using the preloaded model
def classify_image(img_array):
    start_time = time.time()
    predictions = model(img_array)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    print(f"Predictions: {predictions}")  # Debug: Print predictions
    class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    print(f"Class Index: {class_idx}")  # Debug: Print class index
    return class_idx

# Define functions for processing the images based on classification
def process_document(image_path):
    print("Processing document:")
    a1c_value = text_reco(image_path)
    print(a1c_value)
    Diabetes(a1c_value)

def process_MRI(image_path):
    print("Processing MRI:")
    predict_alzh.main(image_path)

def process_lung_image(image_path):
    print("Processing lung image:")
    lung.main(image_path)

def process_malaria_cell(image_path):
    print("Processing malaria cell:")
    malaria.main(image_path)

def process_skin_image(image_path):
    print("Processing skin image:")
    skin_final.main(image_path)

# Centralized function to dispatch processing based on the class
def process_based_on_class(image_path, class_idx):
    if class_idx == 0:
        process_skin_image(image_path)  # skin
    elif class_idx == 1:
        process_document(image_path)  # document
    elif class_idx == 2:
        process_lung_image(image_path)  # lung
    elif class_idx == 3:
        process_malaria_cell(image_path)  # malaria
    elif class_idx == 4:
        process_MRI(image_path)  # Alzheimer's
    else:
        print("Unknown image category.")

def main(image_path):
    img_array = preprocess_image(image_path)
    class_idx = classify_image(img_array)
    process_based_on_class(image_path, class_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an uploaded image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    main(args.image_path)
