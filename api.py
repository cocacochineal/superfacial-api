from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uuid
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import face_recognition
import uvicorn
import io

#Sequential model.
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
#Getting necessary layers.
from keras.layers import Conv2D             #Two-dimensional convolution layer.
from keras.layers import MaxPooling2D       #Two-dimensional pooling layer.
from keras.layers import Flatten            #Flattening layer.
from keras.layers import Dropout            #Regularization to prevent overfitting.
#Image preprocessing.
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#Operating system interface.
import os
#Plotting library.
import matplotlib.pyplot as plt
#Other libraries.
import numpy as np
import random
import PIL
import pandas as pd
from sklearn.model_selection import train_test_split


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
    options_ = [0 if option=='0.0' else 1 for option in options_]
    num_faces={1: 20,
    2: 20,
    3: 20,
    4: 19,
    5: 20,
    6: 20,
    7: 20,
    8: 19,
    9: 20,
    10: 20,
    11: 19,
    12: 19,
    13: 20,
    14: 20,
    15: 20,
    16: 19,
    17: 19,
    18: 20,
    19: 18,
    20: 20,
    21: 20,
    22: 20,
    23: 20,
    24: 20,
    25: 20,
    26: 20,
    27: 20,
    28: 19,
    29: 19,
    30: 20,
    31: 17,
    32: 20,
    33: 20,
    34: 19,
    35: 20,
    36: 20,
    37: 20,
    38: 20,
    39: 19,
    40: 18,
    41: 20,
    42: 19,
    43: 20,
    44: 20,
    45: 20,
    46: 20,
    47: 18,
    48: 9,
    49: 19,
    50: 20,
    51: 19,
    52: 19}
    
    print(options_)
    global options
    options = options_
    global y
    y=[]
    for i in range(1,53):
        for item in num_faces[i]*[options[i-1]]:
            y.append(item)
    print(y)
      

    #print(int.from_bytes(options,"big"))
    return y

@app.post("/image/")
# def cnn(image: bytes=File(...)):
#     

def something(image: bytes=File(...)):
    print(len(y))
    # i=1
    # like_encode=[]
    # for option in options:
    #     if option!='0.0':
    #         like_encode.append(face_recognition.face_encodings(face_recognition.load_image_file(f"/Users/shan/Desktop/interface_face/{i}.jpg"))[0])
    #     i+=1
    # avg = sum(like_encode)/len(like_encode)
    
    image_object = io.BytesIO(image)
    image_to_test = face_recognition.load_image_file(image_object)
    face_landmarks_list = face_recognition.face_landmarks(image_to_test)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)
    
    global model
    def initialize_model():
        model = Sequential()
        #model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        #model.add(MaxPooling2D(pool_size = (2, 2)))
        #model.add(Flatten())
        model.add(Dense(units = 128, input_dim = 128, activation = 'relu'))
        model.add(Dense(units = 64, activation = 'relu'))
        model.add(Dense(units = 100, activation = 'sigmoid'))
        model.add(Dense(units = 1, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model
    model = initialize_model()
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    return model     
    X = pickle.load(open('cele50_encode', 'rb'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1, shuffle=True)
    print(model.evaluate(X_test_y, y_test_y, verbose=0))
    
    results=[]
    for encode in image_to_test_encoding:
        #face_distances = face_recognition.face_distance([avg], encode)
        prediction=model.predict(encode)[0]
        # if face_distances[0]<0.55:
        #     results.append(1)
        # else:
        #     results.append(0)   
        results.append(prediction)
    print(results)
    return [results, face_landmarks_list]
    if __name__ == "__main__":
        uvicorn.run(app, host='0.0.0.0', port=8888)