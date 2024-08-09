import torch
import cv2
import warnings
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

warnings.filterwarnings('ignore')

num_classes = 4   
model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features  
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load('model/lung.pth'), strict=False)
model.eval()  

transform = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(0)
    
    with torch.no_grad():  
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)
        
    class_idx = predicted.item()
    return class_idx, image

def process_adenocarcinoma(image_path):
    print(f"The given sample is adenocarcinoma.")

def process_carcinoma(image_path):
    print(f"The given sample is carcinoma.")

def process_normal(image_path):
    print(f"The given sample is normal.")

def process_squamous(image_path):
    print(f"The given sample is squamous.")

def main(image_path):
    class_idx, image = classify_image(image_path)
    
    class_labels = {
        0: 'Adenocarcinoma',
        1: 'Carcinoma',
        2: 'Normal',
        3: 'Squamous'
    }
    predicted_class_name = class_labels.get(class_idx, "Unknown")
    
    if class_idx == 0:
        process_adenocarcinoma(image_path)
    elif class_idx == 1:
        process_carcinoma(image_path)
    elif class_idx == 2:
        process_normal(image_path)
    elif class_idx == 3:
        process_squamous(image_path)    
    else:
        print("Unknown image category.")
    
    # Display the image and prediction
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class_name}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Classify lung images.')
    parser.add_argument('image_path', type=str, help='Path to the lung image file')
    args = parser.parse_args()

    main(args.image_path)
