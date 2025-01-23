from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS Middleware
from flowers import flowers
from pizza import pizza
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from all origins. Replace "*" with specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict_flowers")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Call the `flowers` function for prediction
        class_name, confidence_score = flowers(file.file)

        # Return the prediction and confidence score
        return JSONResponse(content={
            "class_name": class_name,
            "confidence_score": confidence_score
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/predict_pizza")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Call the `flowers` function for prediction
        class_name, confidence_score = pizza(file.file)

        # Return the prediction and confidence score
        return JSONResponse(content={
            "class_name": class_name,
            "confidence_score": confidence_score
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/")
def root():
    return {"message": "Welcome to the image classification API!"}


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
