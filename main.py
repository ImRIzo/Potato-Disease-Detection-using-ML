from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as TheFuck

app = FastAPI()
MODEL = TheFuck.keras.models.load_model("model/model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def ping():
	return "lol"

def read_file_as_image(data) -> np.ndarray:
	return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(file: UploadFile):
	#bytes = await file.read()
	image = read_file_as_image(await file.read())
	image = np.expand_dims(image,0)
	predictions = MODEL.predict(image) 
	predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
	confidence = float(np.max(predictions[0]))
	confidence =  round(confidence, 2) * 100
	return {
		'class':predicted_class,
		'confidence':confidence
	}

if __name__ == '__main__':
	uvicorn.run(app)