import cv2
import pytesseract
import re
from pytesseract import Output

# def resize_image_keep_aspect(image_path, new_width=None, new_height=None):
    #Load the image
    # image = cv2.imread(image_path)
    # 
    #Check if image was successfully loaded
    # if image is None:
        # print("Error: Image not found.")
        # return None
    # 
    #Get the current dimensions of the image
    # height, width = image.shape[:2]
# 
    #Initialize the scaling factor
    # scaling_factor = 1
# 
    #Calculate the scaling factor based on the desired new width or new height
    # if new_width is not None and new_height is None:
        # scaling_factor = new_width / width
    # elif new_height is not None and new_width is None:
        # scaling_factor = new_height / height
    # elif new_width is not None and new_height is not None:
        # scaling_factor_w = new_width / width
        # scaling_factor_h = new_height / height
        # scaling_factor = min(scaling_factor_w, scaling_factor_h)
# 
    # Calculate new dimensions
    # new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
    # 
    # Resize the image
    # resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    # 
    # return resized_image

# Usage
# a = '/home/cycobot/Documents/bdsm/test/document/document9.jpeg'
# new_width = 1080  # For example, you want the new width to be 800 pixels
# new_height = 500  # For example, you want the new height to be 600 pixels
# resized_image = resize_image_keep_aspect(a, new_width=new_width, new_height=new_height)

# You can now use `resized_image` for further processing, displaying, or saving it to a file


# Load the image
def text_reco(a):
    image_path = a
    image = cv2.imread(image_path)
    
    # Preprocessing the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding to get a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Now apply Tesseract OCR on the preprocessed image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(opening, config=custom_config)
    # print(text)
    match = re.search(r"C\s+(\d+\.\d+)", text, re.IGNORECASE)
    
    if match:
        return match.group(1)
    else:
        print("No A1C value found.")

# text_reco(a)