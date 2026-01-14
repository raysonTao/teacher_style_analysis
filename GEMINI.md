# GEMINI.md - Teacher Style Analysis Project

## Project Overview

This project is a sophisticated system for analyzing teaching styles from video recordings of classroom sessions. It employs a multi-modal deep learning approach, processing video, audio, and text to classify a teacher's style into one of seven categories:

1.  **理论讲授型** (Theory-based)
2.  **启发引导型** (Inspirational)
3.  **互动导向型** (Interactive)
4.  **逻辑推导型** (Logical)
5.  **题目驱动型** (Problem-driven)
6.  **情感表达型** (Emotional)
7.  **耐心细致型** (Patient)

The core of the system is a Multi-Modal Attention Network (MMAN), a deep learning model that combines features from different modalities to make its classification. The project also includes a fascinating component for data annotation that uses a Vision Language Model (VLM) from Anthropic (Claude) to automatically label teaching styles from video data.

The project is structured as a Python application with a command-line interface (CLI) for analyzing videos, training models, and serving an API.

## Building and Running

### 1. Installation

The project's dependencies are listed in `src/requirements.txt`. To install them, run:

```bash
pip install -r src/requirements.txt
```

### 2. Analyzing a Single Video

To analyze a single video and classify the teacher's style, use the `analyze` command:

```bash
python -m src.main analyze \
    --video path/to/your/video.mp4 \
    --teacher "Teacher Name" \
    --discipline "Subject" \
    --grade "Grade Level" \
    --mode deep_learning \
    --device cuda
```

### 3. Training the Model

The deep learning model can be trained using the `train` script. This script can use either a real dataset or generate synthetic data for testing purposes.

```bash
python -m src.models.deep_learning.train \
    --data_path /path/to/your/dataset.json \
    --model_config default \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --device cuda
```

### 4. Running the API Server

The project includes a FastAPI-based API server. To run it, use the `server` command:

```bash
python -m src.main server --host 0.0.0.0 --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`.

## Development Conventions

*   **Configuration:** The project is configured through `src/config/config.py`, which defines paths, model parameters, and other settings.
*   **Modularity:** The code is well-structured and modular, with clear separation of concerns for feature extraction, modeling, training, and API handling.
*   **Entry Point:** The main entry point for the application is `src/main.py`, which uses `argparse` to provide a CLI.
*   **Deep Learning Model:** The core deep learning model is `src/models/deep_learning/mman_model.py`, which uses PyTorch and a combination of Transformers and LSTMs.
*   **Data Annotation:** The project has an innovative data annotation pipeline using a Vision Language Model (VLM) in `src/annotation/vlm_annotator.py`.
*   **Testing:** The project appears to have some tests, as indicated by the `src/tests` directory and `pytest` in the requirements. To run the tests, you can likely use:
    ```bash
    python -m src.tests.run_tests
    ```

This `GEMINI.md` file provides a good starting point for understanding and interacting with this project in the future.
