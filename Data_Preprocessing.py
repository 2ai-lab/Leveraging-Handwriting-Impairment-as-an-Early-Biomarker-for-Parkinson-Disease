# import cv2
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# def preprocess_data(x, y, target_size=(229, 229)):
#     processed_x = []
#     for img in x:
       
#         img = cv2.resize(img, target_size)
       
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         img = cv2.equalizeHist(img)
        
       
#         processed_x.append(img)

   
#     processed_x = np.array(processed_x)
#     processed_x = processed_x / 255.0

#     label_encoder = LabelEncoder()
#     processed_y = label_encoder.fit_transform(y)

#     return processed_x, processed_y

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(x, y, target_size=(229, 229)):
    processed_x = []
    for img in x:
        img = cv2.resize(img, target_size)
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        img = cv2.equalizeHist(img)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        processed_x.append(img)

    processed_x = np.array(processed_x)
    processed_x = processed_x / 255.0

    label_encoder = LabelEncoder()
    processed_y = label_encoder.fit_transform(y)

    return processed_x, processed_y

