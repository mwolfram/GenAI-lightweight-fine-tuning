#!/usr/bin/env python
# coding: utf-8

from typing import Any
from datasets import load_dataset

DEFAULT_SEED = 42
IMDB_DATASET_NAME = "imdb"
SMS_SPAM_DATASET_NAME = "sms_spam"
SPLITS = ["train", "test"]

def imdb():
    return DatasetWrapper(IMDB_DATASET_NAME)

def sms_spam():
    return DatasetWrapper(SMS_SPAM_DATASET_NAME)

class DatasetWrapper():
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.tokenized_dataset = {}
        self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == IMDB_DATASET_NAME:
            self.dataset = {split: ds for split, ds in zip(SPLITS, load_dataset(self.dataset_name, split=SPLITS))}
        elif self.dataset_name == SMS_SPAM_DATASET_NAME:
            self.dataset = load_dataset(self.dataset_name, split="train").train_test_split(test_size=0.2, shuffle=True, seed=23)
        return self

    def reduce_to_fraction(self, fraction, seed=DEFAULT_SEED):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].shuffle(seed=seed).select(range(int(len(self.dataset[split]) * fraction)))
        return self

    def get_id2label(self):
        if self.dataset_name == IMDB_DATASET_NAME:
            return {0: "NEGATIVE", 1: "POSITIVE"}
        elif self.dataset_name == SMS_SPAM_DATASET_NAME:
            return {0: "NOT SPAM", 1: "SPAM"}
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def get_label2id(self):
        if self.dataset_name == IMDB_DATASET_NAME:
            return {"NEGATIVE": 0, "POSITIVE": 1}
        elif self.dataset_name == SMS_SPAM_DATASET_NAME:
            return {"NOT SPAM": 0, "SPAM": 1}
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def get_split_names(self):
        return self.dataset.keys()
    
    def get_dataset(self):
        return self.dataset
    
    def get_train_dataset(self):
        return self.dataset["train"]
    
    def get_test_dataset(self):
        return self.dataset["test"]

    def get_tokenized_dataset(self):
        return self.tokenized_dataset
    
    def get_tokenized_train_dataset(self):
        return self.tokenized_dataset["train"]
    
    def get_tokenized_test_dataset(self):
        return self.tokenized_dataset["test"]

    def print_dataset_info(self):
        for split_name, dataset_split in self.dataset.items():
            print(f"{split_name.capitalize()} dataset:")
            print(dataset_split)
            print()

    def get_text_column_name(self):
        if self.dataset_name == IMDB_DATASET_NAME:
            return "text"
        elif self.dataset_name == SMS_SPAM_DATASET_NAME:
            return "sms"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
if __name__ == "__main__":
    print("SMS Spam Dataset:")
    sms_spam_dsw = DatasetWrapper(dataset_name=SMS_SPAM_DATASET_NAME)
    sms_spam_dsw.reduce_to_fraction(0.1).print_dataset_info()

    print("IMDB Dataset:")
    imdb_dsw = DatasetWrapper(dataset_name=IMDB_DATASET_NAME)
    imdb_dsw.reduce_to_fraction(0.1).print_dataset_info()
