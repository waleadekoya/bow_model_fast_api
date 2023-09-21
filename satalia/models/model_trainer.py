import random

import torch


class Trainer:
    def __init__(self, model, word_to_index, tag_to_index, device="cpu"):
        self.model = model.to(device)
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.tensor_type = torch.LongTensor

        if torch.cuda.is_available():
            self.model.to(device)
            self.tensor_type = torch.cuda.LongTensor

    def train_bow(self, train_data, test_data, epochs=10):
        for ITER in range(epochs):
            # perform training
            self.model.train()
            random.shuffle(train_data)
            total_loss = 0.0
            train_correct = 0
            for sentence, tag in train_data:
                sentence_tensor = torch.tensor(sentence).type(self.tensor_type)
                tag_tensor = torch.tensor([tag]).type(self.tensor_type)

                output = self.model(sentence_tensor)
                predicted = torch.argmax(output.data.detach()).item()

                loss = self.criterion(output, tag_tensor)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if predicted == tag:
                    train_correct += 1

            # perform testing of the model
            self.model.eval()
            test_correct = 0
            for sentence, tag in test_data:
                sentence_tensor = torch.tensor(sentence).type(self.tensor_type)
                output = self.model(sentence_tensor)
                predicted = torch.argmax(output.data.detach()).item()
                if predicted == tag:
                    test_correct += 1

            # print model performance results
            log = f'ITER: {ITER + 1} | ' \
                  f'train loss/sent: {total_loss / len(train_data):.4f} | ' \
                  f'train accuracy: {train_correct / len(train_data):.4f} | ' \
                  f'test accuracy: {test_correct / len(test_data):.4f}'
            print(log)
