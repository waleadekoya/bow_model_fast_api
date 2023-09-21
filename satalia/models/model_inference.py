import torch


class BoWInference:
    def __init__(self, model, word_to_index, tag_to_index, device="cpu"):
        self.model = model
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index
        self.device = device

    def sentence_to_tensor(self, sentence):
        """
        Convert a sentence into a tensor using a word-to-index dictionary.

        Args:
            sentence (str): The input sentence.

        Returns:
            torch.Tensor: The tensor representation of the sentence.
        """
        return torch.tensor([self.word_to_index.get(word, self.word_to_index["<unk>"]) for word in sentence.split(" ")])

    def perform_inference(self, sentence):
        """
        Perform inference on the trained BoW model.

        Args:
            sentence (str): The input sentence for inference.

        Returns:
            str: The predicted class/tag for the input sentence.
        """
        # Preprocess the input sentence to match the model's input format
        sentence_tensor = self.sentence_to_tensor(sentence)

        # Move the input tensor to the same device as the model
        sentence_tensor = sentence_tensor.to(self.device)

        # Make sure the model is in evaluation mode and on the correct device
        self.model.eval()
        self.model.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(sentence_tensor)

        # Move the output tensor to CPU if it's on CUDA
        if self.device == "cuda":
            output = output.cpu()

        # Convert the model's output to a predicted class/tag
        predicted_class = torch.argmax(output).item()

        # Reverse lookup to get the tag corresponding to the predicted class
        for tag, index in self.tag_to_index.items():
            if index == predicted_class:
                return tag

        # Return an error message if the tag is not found
        return "Tag not found"
