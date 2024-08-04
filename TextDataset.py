import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        # Tokenize the input text
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Return input IDs and target as tensors
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove the extra dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Remove the extra dimension
            'target': torch.tensor(target, dtype=torch.long)  # Ensure target is a tensor
        }