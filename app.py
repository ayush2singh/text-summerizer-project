from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline
import uvicorn
import os

app = FastAPI(title="Pegasus Summarizer API")

# Setup template directory to point to your 'templates/' folder
templates = Jinja2Templates(directory="templates")

# 1. INITIALIZE MODEL ONCE AT SERVER STARTUP
# This prevents the 2.28 GB model from reloading on every single button click!
print("Loading Pegasus Summarization Pipeline into memory...")
predictor = PredictionPipeline()
print("Pipeline Ready!")

# 2. DEFINE STRUCTURED JSON REQUEST BODY
class DialogueRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the custom HTML frontend UI on the root URL."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    """Triggers the model training pipeline."""
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(request: DialogueRequest):
    """Accepts JSON payloads from the frontend and returns abstractive summaries."""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Dialogue text cannot be empty.")
    
    try:
        # 3. USE PRE-LOADED MODEL AND ACCESS TEXT FROM JSON BODY
        summary = predictor.predict(request.text)
        return {"status": "success", "summary": summary}
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)