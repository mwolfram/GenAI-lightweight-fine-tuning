#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer

from DatasetWrapper import DatasetWrapper

GPT2_TOKENIZER_NAME = "gpt2"

class TokenizerWrapper():

    def __init__(self, tokenizer_name):
        self.tokenizer_name = tokenizer_name
        self.load_tokenizer()


    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # For GPT-2, the pad token is the eos token
        if self.tokenizer_name == GPT2_TOKENIZER_NAME:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def get_tokenizer(self):
        return self.tokenizer
    

    def tokenize(self, dsw: DatasetWrapper):
        tokenized_dataset = dsw.get_tokenized_dataset()
        for split_name in dsw.get_split_names():
            tokenized_dataset[split_name] = dsw.get_dataset()[split_name].map(
                lambda x: {"input_ids": self.tokenizer(x[dsw.get_text_column_name()], truncation=True, padding="max_length")["input_ids"], "labels": x["label"]}, batched=True
            )


if __name__ == "__main__":
    tokenizer = TokenizerWrapper(GPT2_TOKENIZER_NAME)
    dsw = DatasetWrapper("sms_spam").reduce_to_fraction(0.1)
    tokenizer.tokenize(dsw)
    print(dsw.get_tokenized_dataset())
    print(dsw.get_tokenized_train_dataset()[0])

