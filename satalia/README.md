
# FastAPI Application for BoW Model Training and Inference

## Overview
This is a FastAPI application designed to facilitate training and inference tasks for a Bag-of-Words (BoW) model. It provides two primary endpoints:

1. `/train`: For training the BoW model.
2. `/infer`: For performing inference with the trained model.

## Prerequisites
- Docker

## Setup & Running

1. **Clone the repository**:
    ```
    git clone [your_repository_url]
    cd [repository_directory]
    ```

2. **Build the Docker image**:
    ```
    docker build -t bow_fastapi_image .
    ```

3. **Run the Docker container**:
    ```
    docker run -p 80:80 bow_fastapi_image
    ```

Once the container is running, the FastAPI application will be accessible at `http://localhost:80`.

## Endpoints

### 1. Train the Model

- **Endpoint**: `/train`
- **Method**: `POST`
- **Body**:
    ```json
    {
        "data": [[sentence, label], ...]
    }
    ```
- **Response**: A status indicating the completion of the training.

### 2. Perform Inference

- **Endpoint**: `/infer`
- **Method**: `POST`
- **Body**:
    ```json
    {
        "sentence": "your_sentence_here"
    }
    ```
- **Response**: The predicted label for the input sentence.

## Model & Data Handling
The application uses a Bag-of-Words (BoW) model for classification. The data is sourced from external URLs and is prepared using the `DataHandler` class. The model training and inference are facilitated by the `Trainer` and `BoWInference` classes, respectively.

## Device Support
The application is designed to run on both CPU and GPU (CUDA), depending on the availability.
