from fastapi import FastAPI, HTTPException,UploadFile,File
from pydantic import BaseModel
import pandas as pd
from io import StringIO,BytesIO
from model_caller_function import generate_and_detect,vae
import torch
from sklearn.preprocessing import MinMaxScaler
import base64
from fastapi.responses import StreamingResponse

# Assuming the existence of a function generate_and_detect and a VAE model
# These need to be implemented according to your specific requirements

app = FastAPI()
seq_length = 10
class GenerateRequest(BaseModel):
    num_samples: int
    new_data_json: str

class GenerateResponse(BaseModel):
    synthetic_data: dict
    anomalies: list
    tsne_image_url: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_synthetic(num_of_samples : int,new_data : UploadFile = File(...)):
    try:
        # Convert JSON string to DataFrame
        #new_data = pd.read_json(new_data, orient='records')
        contents =  await new_data.read()
        # Use StringIO to read the contents as if it were a file
        new_data = pd.read_csv(StringIO(contents.decode('utf-8')))
        new_data = new_data.set_index("Date")
        scaler = MinMaxScaler()
        new_data = pd.DataFrame(scaler.fit_transform(new_data), columns=new_data.columns, index=new_data.index)

        # Assuming generate_and_detect is an async function or adapted to work with FastAPI
        synthetic_data, anomalies,img = generate_and_detect(vae, new_data, seq_length, batch_size=num_of_samples, threshold_percentile=95)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        # Convert synthetic_data to dict if it's not already
        synthetic_data_dict = synthetic_data.to_dict(orient='dict') if isinstance(synthetic_data, pd.DataFrame) else synthetic_data
        with open("tsne_plot.png", "wb") as f:
            f.write(img.getvalue())
        return {"synthetic_data": synthetic_data_dict,'anomalies':anomalies,'tsne_image_url' :"tsne_plot.png"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("tsne_plot.png")
async def get_tsne_plot():
    with open("tsne_plot.png", "rb") as f:
        return StreamingResponse(BytesIO(f.read()), media_type="image/png")
