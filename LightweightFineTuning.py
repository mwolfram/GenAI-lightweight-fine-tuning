#!/usr/bin/env python
# coding: utf-8

from Utils import check_versions, enable_cuda
from TokenizerWrapper import TokenizerWrapper
from DatasetWrapper import sms_spam, imdb
from TrainerWrapper import TrainerWrapper

import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# tokenizer
TOKENIZER_NAME = "facebook/opt-350m"
#TOKENIZER_NAME = "distilbert-base-uncased"
#TOKENIZER_NAME = "gpt2"

# model is the same as tokenizer
MODEL_NAME = TOKENIZER_NAME

# downsize the data
FRACTION = 1.0


def train():
    data = sms_spam().reduce_to_fraction(FRACTION)
    tokenizer_wrapper = TokenizerWrapper(TOKENIZER_NAME)
    trainer_wrapper = TrainerWrapper(MODEL_NAME)

    tokenizer_wrapper.tokenize(data)
    
    # Evaluate initial model
    #trainer_wrapper.init_trainer(data, tokenizer_wrapper.get_tokenizer(), lora=False)
    #trainer_wrapper.evaluate()

    # Train Lora model
    #trainer_wrapper.train_with_own_loop(data, lora=True)

    # Evaluate saved Lora model
    model_path = "mwolfram/facebook/opt-350m-lora"
    model = trainer_wrapper.load_peft_model(model_path)
    trainer_wrapper.evaluate_with_own_loop(data, model)

    # Generate text from saved model
    #trainer_wrapper.generate_from_saved_model(tokenizer_wrapper.get_tokenizer(), model_path)


if __name__ == "__main__":
    enable_cuda()
    check_versions()
    train()
