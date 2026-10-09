## Live demo

- [Try the waste classifier](https://waste-image-classifier-yiy6.onrender.com/)
- [Interactive API documentation](https://waste-image-classifier-yiy6.onrender.com/docs)

Hosted on Render using Docker. The free service sleeps after
inactivity, so the first request may take longer while it wakes up.

Predictions can be incorrect, even when the model score is high.


# Waste Image Classifier

An end-to-end machine learning project that classifies waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The project compares a classical machine learning approach, a CNN trained from scratch, and transfer learning with MobileNetV2. The selected model is served through FastAPI with a browser-based image upload interface.

## Project overview

This project covers the complete workflow from inspecting image data to serving predictions:

1. Inspect the dataset and identify data quality issues.
2. Create reproducible, group-aware training, validation, and test splits.
3. Extract HOG and color features and train an SVM baseline.
4. Tune the SVM using validation macro F1.
5. Train a CNN from scratch.
6. Train a classifier on top of a frozen pretrained MobileNetV2.
7. Fine-tune selected MobileNetV2 layers.
8. Select the model using validation results.
9. Evaluate the selected model on the held-out test set.
10. Serve predictions through a FastAPI application.
11. Add an image upload page with prediction scores and error handling.

## Results at a glance

The selected model is a fine-tuned MobileNetV2.

| Final test metric | Result |
|---|---:|
| Test images | 377 |
| Correct predictions | 318 |
| Accuracy | 84.35% |
| Macro F1 | 0.8252 |

These results describe performance on the project's held-out test split. They do not guarantee the same performance on arbitrary real-world images.

## Technology stack

| Area | Tools |
|---|---|
| Programming language | Python 3.11 |
| Deep learning | TensorFlow 2.21.0 and Keras |
| Classical machine learning | scikit-learn |
| Image feature extraction | scikit-image |
| Image validation | Pillow |
| Numerical operations | NumPy |
| Visualization | Matplotlib |
| API | FastAPI |
| Application server | Uvicorn |
| File uploads | python-multipart |
| Frontend | HTML, CSS, and JavaScript |
| Development | VS Code and Jupyter Notebook |
| Version control | Git and GitHub |

The recorded deep learning experiments ran locally on CPU.

## Dataset

The project uses images from the TrashNet dataset.

Dataset repository: https://github.com/garythung/trashnet

Refer to the original repository for dataset documentation and licensing information.

### Original class distribution

| Class | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

The dataset is imbalanced. For example, the paper class contains substantially more images than the trash class.

### Data inspection

Before training, the dataset was inspected for:

- Class counts.
- Image dimensions and readability.
- Exact duplicate images.
- Similar or related images.
- Potentially conflicting labels.
- Representative images from each category.

SHA-256 hashes were used to identify exact duplicate content. Similarity checks helped identify candidates for visual review.

Following the project's exclusions, 2,521 images remained for splitting.

### Reproducible dataset splits

The retained images were divided into approximately:

- 70% training.
- 15% validation.
- 15% testing.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| Cardboard | 282 | 61 | 60 |
| Glass | 348 | 75 | 75 |
| Metal | 286 | 62 | 61 |
| Paper | 416 | 89 | 89 |
| Plastic | 336 | 72 | 72 |
| Trash | 96 | 21 | 20 |
| **Total** | **1,764** | **380** | **377** |

Group-aware splitting kept identified related images together. Exact duplicate hashes and assigned groups were checked for overlap across splits.

This reduces the risk of evaluating a model on images that are duplicates or close relatives of its training examples. It does not guarantee that every possible similarity in the dataset was detected.

### CSV manifests

The split assignments are recorded in:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Each manifest contains:

| Column | Purpose |
|---|---|
| `path` | Image path relative to the project root |
| `label` | Human-readable class name |
| `label_id` | Integer class identifier |
| `group_id` | Identifier used to keep related images in the same split |
| `sha256` | Image content hash used for duplicate checks |

The models use these recorded splits instead of independently creating new random splits for each experiment.

### Class mapping

The class order is:

| Label ID | Class |
|---|---|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

Inference reads the class names from the selected model's metadata so that output scores are interpreted in the correct order.

## Model development

### 1. HOG and color SVM baseline

The classical machine learning pipeline uses manually extracted image features.

**HOG features**

Histogram of Oriented Gradients describes local edge patterns and shapes.

- HOG features per image: 1,764.

**Color features**

Color histograms describe the distribution of pixel values across the three color channels.

- 32 bins per channel.
- Three channels.
- Color features per image: 96.

**Combined feature vector**

Each image is represented by:

1,764 HOG features + 96 color features = 1,860 features.

The classification pipeline applies feature standardization followed by an SVM with an RBF kernel.

The baseline configuration used:

- `C=1.0`
- `gamma="scale"`
- `class_weight="balanced"`

### 2. SVM tuning

Nine configurations were compared using the same validation split.

| Parameter | Values |
|---|---|
| `C` | 0.1, 1.0, 10.0 |
| `gamma` | `"scale"`, 0.0001, 0.00001 |

The selected configuration used:

- `C=10.0`
- `gamma="scale"`

Validation macro F1 improved from 0.6653 to 0.7085.

### 3. CNN trained from scratch

A small CNN was trained without pretrained weights.

The architecture included:

- Input images of shape `128 × 128 × 3`.
- Pixel rescaling.
- Random horizontal flips.
- Random rotations.
- Random zoom.
- Three convolutional blocks with 16, 32, and 64 filters.
- Max pooling.
- Global average pooling.
- A dense layer with 64 units.
- Dropout.
- A six-unit softmax output layer.

The CNN learned its image features directly from the training images.

The longer scratch-CNN experiment achieved:

- Validation accuracy: 58.16%.
- Validation macro F1: 0.5610.

It performed below the tuned SVM and the pretrained MobileNetV2 models on this split.

### 4. Frozen MobileNetV2

MobileNetV2 was initialized with ImageNet pretrained weights.

Its original classification head was removed and replaced with:

- Global average pooling.
- Dropout.
- A dense layer with six softmax outputs.

During this stage, the pretrained base was frozen. Only the new classification head learned from the waste images.

Configuration:

| Setting | Value |
|---|---|
| Input size | 128 × 128 × 3 |
| Pretrained weights | ImageNet |
| Base model trainable | No |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Selection metric | Validation macro F1 |

The frozen model achieved:

- Validation accuracy: 80.00%.
- Validation macro F1: 0.7829.

### 5. Fine-tuned MobileNetV2

Fine-tuning started from the saved frozen-model checkpoint.

The later MobileNetV2 layers were allowed to update so that their features could adapt to the waste classification task.

Configuration:

| Setting | Value |
|---|---|
| Starting checkpoint | `mobilenet_v2_frozen.keras` |
| Fine-tuning begins at | `block_13_expand` |
| Earlier base layers | Frozen |
| Batch normalization layers | Frozen |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Maximum fine-tuning epochs | 15 |
| Best recorded epoch | 15 |
| Selection metric | Validation macro F1 |

The fine-tuned model achieved:

- Training accuracy: 97.00%.
- Validation accuracy: 82.63%.
- Validation macro F1: 0.8099.
- Macro F1 improvement over the frozen model: 0.0270.

The lower learning rate allowed gradual adjustments to pretrained features.

## Training data pipeline

The shared CNN data-loading module is `training/cnn_data.py`.

It:

1. Reads the requested split manifest.
2. Resolves image paths.
3. Loads and decodes images as three-channel RGB.
4. Resizes images to 128 × 128 with antialiasing.
5. Converts image tensors to `float32`.
6. Groups examples into batches.
7. Prefetches batches for model execution.

Configuration:

- Image size: 128 × 128.
- Batch size: 32.
- Main random seed: 42.

### Pixel preprocessing

The data loader supplies pixels on approximately the 0–255 scale.

Normalization is part of the model:

- Scratch CNN: divides pixel values by 255.
- MobileNetV2: transforms pixels using `pixel / 127.5 - 1`.

The API also supplies pixels on the 0–255 scale. It does not apply MobileNetV2 normalization a second time.

### Data augmentation

Random flips, rotations, and zooms create variations of the training images.

Augmentation is active during training. It is inactive during validation, testing, and API prediction.

### Training order and evaluation order

Training examples are shuffled to vary their order between epochs.

Validation and test examples retain manifest order because evaluation compares model predictions with separately stored labels in that same order.

There is no special requirement that evaluation images be sorted by class. The important requirement is that each prediction matches its corresponding actual label.

### Class imbalance

Balanced class weights were used during neural-network training to increase the contribution of underrepresented classes to the training loss.

Macro F1 was used for model selection because it gives each class equal weight in the final average.

## Validation model comparison

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| HOG + color SVM baseline | 65.79% | 0.6653 |
| Tuned HOG + color SVM | 70.26% | 0.7085 |
| CNN from scratch, longer run | 58.16% | 0.5610 |
| Frozen MobileNetV2 | 80.00% | 0.7829 |
| Fine-tuned MobileNetV2 | **82.63%** | **0.8099** |

The fine-tuned MobileNetV2 was selected using validation macro F1.

Validation data guided checkpoint selection and training decisions. The test set was reserved for final evaluation.

## Final test evaluation

The selected fine-tuned MobileNetV2 was evaluated on 377 held-out test images.

| Metric | Result |
|---|---:|
| Correct predictions | 318 / 377 |
| Accuracy | 0.8435 |
| Macro precision | 0.838 |
| Macro recall | 0.827 |
| Macro F1 | 0.8252 |
| Weighted F1 | 0.845 |

### Per-class test results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cardboard | 0.869 | 0.883 | 0.876 | 60 |
| Glass | 0.967 | 0.787 | 0.868 | 75 |
| Metal | 0.674 | 0.984 | 0.800 | 61 |
| Paper | 0.900 | 0.809 | 0.852 | 89 |
| Plastic | 0.897 | 0.847 | 0.871 | 72 |
| Trash | 0.722 | 0.650 | 0.684 | 20 |

### Observations

- Metal recall was high, but its lower precision indicates that some other materials were incorrectly predicted as metal.
- Trash had the lowest per-class F1 and the fewest test examples.
- Performance varied across classes, which is why accuracy alone was not used to assess the model.

The final test results are retained as reported. Further model development should use training and validation data rather than repeatedly tuning against this test set.

## Application architecture

The application contains three main parts:

| Component | Responsibility |
|---|---|
| Browser interface | Select an image, send it to the API, and display results |
| FastAPI application | Handle routes, upload limits, model availability, and HTTP responses |
| Inference module | Validate image content, preprocess it, run the model, and format scores |

### Prediction request flow

1. The user selects an image in the browser.
2. JavaScript sends a multipart file upload to `/predict`.
3. The API checks model availability and uploaded file size.
4. The inference module validates the image.
5. TensorFlow decodes and resizes the image.
6. The saved model produces six class scores.
7. The highest-scoring class becomes the prediction.
8. FastAPI returns a JSON response.
9. The browser displays the prediction and class scores.

### Model loading

The model is loaded during application startup and reused for subsequent requests.

It is loaded once per server process. Restarting the server or triggering a development reload loads it again.

Startup also checks that the model's input and output shapes match the metadata.

### Prediction execution

The prediction endpoint uses a regular `def` function.

Image processing and TensorFlow inference are synchronous operations in this implementation. A lock serializes access to the model's inference call within each process.

The application does not retrain the model when an image is uploaded.

## Project structure

Important project files and directories:

```text
waste-image-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── inference.py
│   └── static/
│       └── index.html
├── training/
│   ├── prepare_data.py
│   ├── train_svm.py
│   ├── tune_svm.py
│   ├── cnn_data.py
│   ├── train_mobilenet.py
│   ├── finetune_mobilenet.py
│   └── evaluate_test.py
├── data/
│   ├── raw/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── split_summary.json
├── artifacts/
│   ├── mobilenet_v2_frozen.keras
│   └── mobilenet_v2_finetuned.keras
├── reports/
│   ├── dataset_inspection.json
│   ├── svm_baseline_metrics.json
│   ├── svm_tuning_results.json
│   ├── mobilenet_v2_frozen_metrics.json
│   ├── mobilenet_v2_finetuned_metrics.json
│   └── mobilenet_v2_test_metrics.json
├── notebooks/
├── shared/
├── tests/
├── requirements-app.txt
├── requirements-train.txt
└── README.md
```

This listing highlights the main files and is not an exhaustive inventory.

Dataset images and generated model files may need to be obtained or generated separately; their appearance in this structure does not imply that they are committed to Git.

### Folder responsibilities

| Folder | Responsibility |
|---|---|
| `app/` | API, inference, and upload interface |
| `training/` | Data preparation, feature extraction, training, and evaluation |
| `data/raw/` | Local dataset images |
| `data/splits/` | Recorded split assignments and summary |
| `artifacts/` | Saved model files |
| `reports/` | Metrics, metadata, and evaluation outputs |
| `notebooks/` | Exploratory analysis and visualizations |
| `shared/` | Reserved for reusable utilities |
| `tests/` | Reserved for automated checks |

## Windows local setup

Run commands in Command Prompt from the project root.

### 1. Clone the repository

```bat
git clone https://github.com/RariRaj/waste-image-classifier.git
cd waste-image-classifier
```

If the project is already on your computer, open its existing folder instead.

### 2. Check Python

```bat
py -3.11 --version
```

### 3. Create and activate a virtual environment

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

For an existing environment, run only the activation command.

### 4. Install application dependencies

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

### 5. Confirm the required model files

The API expects:

```text
artifacts/mobilenet_v2_finetuned.keras
reports/mobilenet_v2_finetuned_metrics.json
```

Both files must come from the corresponding selected training run.

The JSON report supplies class names and image-size metadata. The `.keras` file contains the saved model.

Cloning the source code alone is insufficient if these generated files are not included.

### 6. Start the development server

```bat
python -m uvicorn app.main:app --reload
```

Open:

| Page | URL |
|---|---|
| Image upload interface | http://127.0.0.1:8000/ |
| Interactive API documentation | http://127.0.0.1:8000/docs |
| Health endpoint | http://127.0.0.1:8000/health |
| OpenAPI schema | http://127.0.0.1:8000/openapi.json |

Stop the server with `Ctrl+C`.

The `--reload` option is intended for local development.

## Using the upload page

1. Start the server.
2. Open http://127.0.0.1:8000/.
3. Select a JPEG or PNG image.
4. Check the image preview.
5. Click **Classify image**.
6. Read the predicted class and the scores for all six categories.

Open the page through the running server rather than opening the HTML file directly.

## API endpoints

### `GET /`

Returns the image upload webpage.

### `GET /health`

Reports whether the application has a loaded predictor.

Example after successful startup:

```json
{
  "status": "ok",
  "model_ready": true,
  "stage": "model_integration"
}
```

This is an application readiness indicator, not a measurement of prediction quality.

### `POST /predict`

Accepts a multipart upload with the field name `file`.

Example from Windows Command Prompt:

```bat
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@D:\path\to\image.jpg"
```

Replace the example path with the path to an actual image.

The response contains:

| Field | Meaning |
|---|---|
| `predicted_class` | Highest-scoring category |
| `confidence` | Score assigned to that category |
| `class_scores` | Scores for all six categories |
| `model` | Model identifier |

Example response from a manual test:

```json
{
  "predicted_class": "plastic",
  "confidence": 0.7519994378089905,
  "class_scores": {
    "cardboard": 0.0005225774948485196,
    "glass": 0.00031817416311241686,
    "metal": 0.02856636978685856,
    "paper": 0.06537644565105438,
    "plastic": 0.7519994378089905,
    "trash": 0.15321700274944305
  },
  "model": "mobilenet_v2_finetuned"
}
```

The image in this example was intended to depict paper. The incorrect plastic prediction illustrates that a high score does not guarantee correctness.

### Upload validation

The application checks for:

- Empty uploads.
- Files larger than 5 MiB.
- Unreadable or invalid image content.
- Formats other than JPEG and PNG.
- Animated images.
- Images exceeding 12 million pixels.

Validation checks image content rather than trusting only its filename extension.

### Error responses

| HTTP status | Meaning |
|---|---|
| `400` | Empty, invalid, unsupported, or otherwise rejected image |
| `413` | Uploaded file exceeds 5 MiB |
| `422` | Request validation failure, such as a missing required file field |
| `503` | Predictor unavailable when a request is handled |

Example invalid-file response:

```json
{
  "detail": "The file is not a valid readable image."
}
```

The API documentation describes possible responses. The response's `detail` field provides the actual error message for a particular request.

If model loading fails during startup, the application may fail to start instead of serving a `503` response.

## Training and evaluation commands

Install the training dependencies before running training scripts:

```bat
python -m pip install -r requirements-train.txt
```

The raw dataset and split manifests must be present at the paths expected by the scripts.

Run commands from the project root.

### Check the CNN data pipeline

```bat
python -m training.cnn_data
```

This prints dataset counts and sample batch information.

### Train the SVM baseline

```bat
python -m training.train_svm
```

### Tune the SVM

```bat
python -m training.tune_svm
```

### Train the frozen MobileNetV2 classifier

```bat
python -m training.train_mobilenet
```

The initial use of ImageNet weights may require downloading them.

### Fine-tune MobileNetV2

```bat
python -m training.finetune_mobilenet
```

This requires the frozen-model checkpoint.

### Evaluate the selected model on the test set

```bat
python -m training.evaluate_test
```

This command documents the final evaluation procedure. Do not repeatedly use test results to choose model changes.

Training scripts may replace their corresponding generated checkpoints and reports. Preserve existing results before starting a new experiment if you need to compare runs.

## Reproducibility

The project records:

- Split membership in CSV manifests.
- Image hashes and group identifiers.
- A main random seed of 42.
- Class names and their order.
- Input image dimensions.
- TensorFlow version.
- Selected epoch.
- Training and validation metrics.
- Training history.
- Classification reports and confusion matrices.

Seeds and recorded splits help reproduce experiments. Exact numerical results can still differ across hardware, software versions, and execution settings.

## Manual checks performed

The application was manually checked for:

- Image upload through the browser interface.
- Image preview and prediction display.
- Class scores returned by the API.
- Rejection of a text file with HTTP `400`.
- Agreement between direct model inference and API inference for a reviewed image.

These checks are not a claim of comprehensive automated test coverage.

## Automated API tests

Install development dependencies and run:

```bat
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_api.py -v
```

The 10 integration tests cover the homepage, model readiness,
JPEG and PNG predictions, invalid content, empty uploads,
oversized uploads, missing files, unsupported formats, and
an unavailable model.

Tests run locally using the saved model. They check API behavior,
not classification accuracy, and do not contact the deployed service.

## Continuous integration and deployment

GitHub Actions runs the 10 API integration tests on pushes to
main and pull requests targeting main.

Render is configured to deploy automatically after CI checks pass.
A failed check prevents automatic deployment of that commit.

The tests use the saved model and generated sample images.
They check API behavior, not model accuracy or production load capacity.

## Known limitations

### Limited dataset size

The model was developed using a relatively small dataset. Some classes contain far fewer examples than others.

### Real-world appearance differences

Uploaded photos can differ from dataset images in:

- Lighting.
- Background clutter.
- Camera angle.
- Object size.
- Material texture.
- Dirt, damage, and deformation.

These differences can reduce prediction quality.

### Confident mistakes

Softmax scores are not a guarantee of correctness or a calibrated estimate of real-world reliability.

An incorrect prediction can still receive a high score.

### No unknown category

The model always selects one of the six supported classes. It has no explicit category for unrelated images or unfamiliar materials.

### One prediction per image

This is an image classifier, not an object detector. A photo containing several waste types receives one overall category.

### Recycling guidance

Material classification does not determine whether an item is recyclable under local rules.

### Deployment scope

The application has been run locally. Containerization, public hosting, and production monitoring are future work.

## Planned improvements

- Add automated API and preprocessing checks.
- Document model artifact retrieval for a fresh installation.
- Build a separate real-world validation collection.
- Investigate recurring mistakes across multiple examples.
- Evaluate confidence calibration and an abstention strategy.
- Measure inference latency and memory usage.
- Add deployment monitoring if publicly hosted.

Any new thresholds or model choices should be developed using appropriate validation data, while preserving the role of the held-out test set.

## Learning outcomes

This project provided practical experience with:

- Dataset inspection and duplicate handling.
- Group-aware splitting and data leakage prevention.
- HOG and color feature extraction.
- SVM training and hyperparameter selection.
- CNN training and augmentation.
- Transfer learning and fine-tuning.
- Handling class imbalance.
- Macro F1 and per-class evaluation.
- Consistent training and inference preprocessing.
- Loading and serving a saved model.
- FastAPI file uploads and error handling.
- Connecting a browser interface to a prediction API.

## References

- TrashNet dataset: https://github.com/garythung/trashnet
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- scikit-learn: https://scikit-learn.org/
- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/