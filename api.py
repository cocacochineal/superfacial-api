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
    print(len(y))
    # i=1
    # like_encode=[]
    # for option in options:
    #     if option!='0.0':
    #         like_encode.append(face_recognition.face_encodings(face_recognition.load_image_file(f"/Users/shan/Desktop/interface_face/{i}.jpg"))[0])
    #     i+=1
    # avg = sum(like_encode)/len(like_encode)
    avg=np.array([-0.05895312,  0.0745253 ,  0.03380865, -0.04731128, -0.08491885,
       -0.02961966, -0.05191092, -0.10862959,  0.14372534, -0.07666109,
        0.27531608, -0.00215914, -0.22524711, -0.10346345, -0.02881772,
        0.10590863, -0.15179179, -0.10992581, -0.01464245, -0.03897626,
        0.09400089,  0.03049925,  0.04361126,  0.02129004, -0.11924487,
       -0.30149831, -0.06317274, -0.0645417 ,  0.06719439, -0.10945869,
        0.0076423 ,  0.05229391, -0.19245138, -0.06429987,  0.02065386,
        0.08652538, -0.07043852, -0.06225016,  0.19108021,  0.03459134,
       -0.14663414,  0.01323044, -0.02724313,  0.27925713,  0.18702968,
        0.03974114,  0.02003187, -0.15192135,  0.10985727, -0.24433079,
        0.07107684,  0.16333052,  0.07042256,  0.06401812,  0.0436044 ,
       -0.15644974,  0.04818465,  0.10099875, -0.19695896,  0.06098775,
        0.06687803, -0.02233723,  0.00497208, -0.08505531,  0.21065185,
        0.05436892, -0.11417659, -0.14432085,  0.11005539, -0.13054259,
       -0.10125365,  0.11344694, -0.13645897, -0.19573336, -0.29942531,
        0.02756927,  0.35478593,  0.12477077, -0.18834226,  0.00852417,
       -0.05529355, -0.00863172,  0.10900245,  0.06525503, -0.03068945,
       -0.02844287, -0.10301634,  0.00854152,  0.19495285, -0.03673798,
       -0.09611909,  0.23435274, -0.00139264,  0.06649028,  0.05265563,
        0.05891484, -0.07191548,  0.03963904, -0.13435482, -0.03143024,
        0.04344594, -0.07645156, -0.00183903,  0.13006889, -0.15507843,
        0.17883338, -0.01178621,  0.01827412,  0.01456133, -0.05132867,
       -0.13738021, -0.00994921,  0.15100468, -0.28005914,  0.26580951,
        0.18423581,  0.10626658,  0.12092329,  0.08839736,  0.04223279,
       -0.02754693, -0.0313828 , -0.16761477, -0.04406595,  0.02360056,
       -0.02829835,  0.05815861,  0.00882281])
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