import tempfile
import os
from fastapi import FastAPI, UploadFile, File

from predict_audio import predict_audio


app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        predicted_label = predict_audio(temp_file_path)

    return {"predicted_label": predicted_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
