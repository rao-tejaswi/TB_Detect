import base64
from fastapi import FastAPI, File
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from base64 import b64encode
from json import dumps, loads
import tensorflow as tf
from PIL import Image
import numpy as np


def get_image_from_bytes(binary_image, max_size=1024):
    image = Image.open(io.BytesIO(binary_image)).convert("RGB").resize((224, 224))
    image = np.asarray(image)
    image = image / 225.0  # Normalize the image
    image = np.float32(image)
    # print(image.shape)
    return image



app = FastAPI(
    title="Tb predictor",
    description="yoo",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')


@app.post("/predict")
async def detect_return_json_result(file: bytes = File(...)):
    image = get_image_from_bytes(file)
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="optimized_model.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image[np.newaxis, ...])

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(np.argmax(output_data))
    # Postprocess and use the results
    class_labels = ["Normal", "TB"]  # Replace with your own class labels
    predicted_class = class_labels[np.argmax(output_data)]

    return {"result" : predicted_class}
