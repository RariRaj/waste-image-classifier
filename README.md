# Waste Image Classifier — Phase 1

This starter establishes the Python environment and a working FastAPI server.
It does not contain a dataset, trained model, or prediction endpoint yet.

## Windows setup (Command Prompt)

Extract the ZIP and open the folder containing this README in VS Code.
Select Terminal > New Terminal and use the Command Prompt profile.
Use Python 3.11 for this project. Check it with `py -3.11 --version`.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m uvicorn app.main:app --reload
```

Run all commands from the folder containing requirements-app.txt.
If Python 3.11 is unavailable, check installed versions with `py -0p`
before proceeding. Do not reuse another project's virtual environment.

Open http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs.
The health response must report status=ok and model_ready=false.
False is expected: model training and loading are later phases.
Stop the development server with Ctrl+C. --reload is for local development.

## Understand the code

- FastAPI() creates the application object.
- @app.get("/health") connects a GET request at /health to health().
- Returning a dictionary produces a JSON response.
- Uvicorn runs the server. app.main:app means the app object in app/main.py.
- /docs is generated from the registered endpoints.

## Folder responsibilities

app/: HTTP API and, later, the image-upload interface.
training/: dataset preparation, training, and evaluation scripts.
shared/: preprocessing shared between training and inference.
data/raw/: original dataset; do not commit image data by default.
data/splits/: reproducible train/validation/test split records.
artifacts/: selected trained model and its label/preprocessing metadata.
reports/: measured metrics and evaluation figures.
notebooks/: exploratory analysis.
tests/: application and preprocessing checks added alongside those features.

## Next milestone

Review the original TrashNet dataset source, license, and image labels.
Inspect class counts, corrupted images, duplicates, and sample photos.
Create reproducible leakage-aware splits before training MobileNetV2.
Install TensorFlow and image/data dependencies when beginning that milestone,
with versions checked against the training environment.

## Reference

https://fastapi.tiangolo.com/tutorial/first-steps/
