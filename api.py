from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uuid
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import face_recognition
import uvicorn
import io

# #Sequential model.
# from keras.models import Sequential
# from keras.layers import Activation, Dense
# #Getting necessary layers.
# from keras.layers import Conv2D             #Two-dimensional convolution layer.
# from keras.layers import MaxPooling2D       #Two-dimensional pooling layer.
# from keras.layers import Flatten            #Flattening layer.
# from keras.layers import Dropout            #Regularization to prevent overfitting.
# #Image preprocessing.
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# #Operating system interface.
# import os
# #Plotting library.
# import matplotlib.pyplot as plt
# #Other libraries.
# import numpy as np
# import random
# import PIL
# import pandas as pd
# from sklearn.model_selection import train_test_split


# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




@app.post('/form')
async def create_item(options_: list = Form(...)):
    #options_object = io.BytesIO(options)
    print(options_)
    global options
    options = options_
    #print(options_object)
      

    #print(int.from_bytes(options,"big"))
    return options_

@app.post("/image/")
# def cnn(image: bytes=File(...)):
#     global model
#     def initialize_model():
#         model = Sequential()
#         #model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#         #model.add(MaxPooling2D(pool_size = (2, 2)))
#         #model.add(Flatten())
#         model.add(Dense(units = 128, input_dim = 129, activation = 'relu'))
#         model.add(Dense(units = 64, activation = 'relu'))
#         model.add(Dense(units = 100, activation = 'sigmoid'))
#         model.add(Dense(units = 1, activation = 'sigmoid'))
#         model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#         return model
#     model = initialize_model()
#     model.compile(loss='binary_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])
#     return model        

def something(image: bytes=File(...)):
    print(len(options))
    i=1
    like_encode=[]
    for option in options:
        if option!='0.0':
            like_encode.append(face_recognition.face_encodings(face_recognition.load_image_file(f"/Users/shan/Desktop/interface_face/{i}.jpg"))[0])
        i+=1
    avg = sum(like_encode)/len(like_encode)
    image_object = io.BytesIO(image)
    image_to_test = face_recognition.load_image_file(image_object)
    face_landmarks_list = face_recognition.face_landmarks(image_to_test)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)
    results=[]
    for encode in image_to_test_encoding:
        face_distances = face_recognition.face_distance([avg], encode)
        # if face_distances[0]<0.55:
        #     results.append(1)
        # else:
        #     results.append(0)   
        results.append(face_distances[0])
    return [results, face_landmarks_list]
    if __name__ == "__main__":
        uvicorn.run(app, host='0.0.0.0', port=8888)