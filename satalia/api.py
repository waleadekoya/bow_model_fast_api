import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.model_trainer import Trainer
from models.model_inference import BoWInference
from data_handler import DataHandler
from models.classifiers import BoWClassifier

app = FastAPI()

# cpu or gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the data_urls variable
data_urls = [
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt",
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt",
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt"
]
# Test the DataHandler class
data_handler = DataHandler(data_urls)
data_handler.prepare_data()

# Display some sample data and vocab

number_of_words = data_handler.number_of_words
number_of_tags = data_handler.number_of_tags
test_data = data_handler.test_data
train_data = data_handler.train_data
word_to_index = data_handler.word_to_index
tag_to_index = data_handler.tag_to_index

model = BoWClassifier(number_of_words, number_of_tags).to(device)


class TrainingData(BaseModel):
    data: list  # Can be more specific depending on data structure


class InferenceRequest(BaseModel):
    sentence: str


@app.post("/train")
def train_model(data: TrainingData):
    trainer = Trainer(model, word_to_index, tag_to_index)
    trainer.train_bow(train_data, test_data, epochs=10)  # Train the model
    return {"status": "training completed"}


@app.post("/infer")
def infer(data: InferenceRequest):
    inference = BoWInference(model, word_to_index, tag_to_index)
    prediction = inference.perform_inference(data.sentence)
    return {"prediction": prediction}
