import torch
from torch.utils.data import Dataset

class Subtask1Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item
    
    def __len__(self):
        return len(self.labels)
