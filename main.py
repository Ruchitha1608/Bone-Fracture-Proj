from fastapi import FastAPI, File, UploadFile, HTTPException
from model import Efficient_net_b0_model
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_model = Efficient_net_b0_model()

@app.get('/healthcheck')
async def healthCheck():
    return {"status":200,"message":'Health Check working'}

@app.post('/predictimage')
async def predictImage(file: UploadFile = File(...)):
    
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as image_file:
            image_file.write(contents)

        prediction = current_model.predictImage('./'+file.filename)

        os.remove(f'./{file.filename}')

        return {'status':200, "prediction":prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
        # print("Error predicting image:",e)
    
