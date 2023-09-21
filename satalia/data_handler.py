import os

import requests


class DataHandler:
    def __init__(self, _data_urls, data_dir='data/classes'):
        self.data_urls = _data_urls
        self.data_dir = data_dir
        self.word_to_index = {"<unk>": 0}
        self.tag_to_index = {}
        self.train_data = []
        self.test_data = []

    def download_data(self):
        # Check if data directory exists, if not create it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Download data files
        for url in self.data_urls:
            filename = os.path.join(self.data_dir, os.path.basename(url))
            if not os.path.exists(filename):
                response = requests.get(url)
                with open(filename, 'wb') as file:
                    file.write(response.content)

    @staticmethod
    def read_data(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.lower().strip()
                line = line.split(' ||| ')
                data.append(line)
        return data

    def load_data(self):
        self.train_data = self.read_data(os.path.join(self.data_dir, 'train.txt'))
        self.test_data = self.read_data(os.path.join(self.data_dir, 'test.txt'))

    def create_vocab(self, data, check_unk=False):
        for line in data:
            for word in line[1].split(" "):
                if not check_unk:
                    if word not in self.word_to_index:
                        self.word_to_index[word] = len(self.word_to_index)
                else:
                    if word not in self.word_to_index:
                        self.word_to_index[word] = self.word_to_index["<unk>"]

            if line[0] not in self.tag_to_index:
                self.tag_to_index[line[0]] = len(self.tag_to_index)

    def create_tensor(self, data):
        for line in data:
            yield [self.word_to_index[word] for word in line[1].split(" ")], self.tag_to_index[line[0]]

    def prepare_data(self):
        self.download_data()
        self.load_data()
        self.create_vocab(self.train_data)
        self.create_vocab(self.test_data)
        self.train_data = list(self.create_tensor(self.train_data))
        self.test_data = list(self.create_tensor(self.test_data))

    @property
    def number_of_words(self):
        return len(self.word_to_index)

    @property
    def number_of_tags(self):
        return len(self.tag_to_index)
