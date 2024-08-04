#!/usr/bin/env python
# coding: utf-8

from Utils import check_versions, enable_cuda
from TokenizerWrapper import TokenizerWrapper
from DatasetWrapper import sms_spam, imdb
from TrainerWrapper import TrainerWrapper

# tokenizer
#TOKENIZER_NAME = "facebook/opt-350m"
#TOKENIZER_NAME = "distilbert-base-uncased"
TOKENIZER_NAME = "gpt2"

# model is the same as tokenizer
MODEL_NAME = TOKENIZER_NAME

# downsize the data
FRACTION = 1.0


def train():
    data = sms_spam().reduce_to_fraction(FRACTION)
    tokenizer_wrapper = TokenizerWrapper(TOKENIZER_NAME)
    trainer_wrapper = TrainerWrapper(MODEL_NAME)

    tokenizer_wrapper.tokenize(data)
    
    # trainer_wrapper.init_trainer(data, tokenizer_wrapper.get_tokenizer(), lora=False)
    # trainer_wrapper.train()
    # trainer_wrapper.evaluate()

    trainer_wrapper.train_with_own_loop(data, lora=True)


if __name__ == "__main__":
    enable_cuda()
    check_versions()
    train()
