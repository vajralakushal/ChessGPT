from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch import *

class AbstractDataHandler(ABC):
    @abstractmethod
    def create_embeddings(self, corpus_file, output_dir):
        pass
    
    @abstractmethod
    def encode_state(self, state):
        pass
    
    @abstractmethod
    def is_valid_token(self, token, state):
        pass
    
    @abstractmethod
    def get_state_size(self):
        pass

class AbstractDataset(Dataset):
    def __init__(self, file_path, vocab, data_handler):
        self.vocab = vocab
        self.data_handler = data_handler
        self.data = []
        self.skipped_items = 0
        self.load_data(file_path)

    @abstractmethod
    def load_data(self, file_path):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        states, tokens = zip(*item)
        return torch.stack(states), torch.tensor(tokens)