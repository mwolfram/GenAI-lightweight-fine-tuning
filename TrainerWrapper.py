#!/usr/bin/env python
# coding: utf-8

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM

from peft import LoraConfig
from peft import get_peft_model

from DatasetWrapper import sms_spam, imdb
from TokenizerWrapper import TokenizerWrapper

class TrainerWrapper():
        
    def __init__(self, model_name):
        self.model_name = model_name
        self.learning_rate = 2e-5
        self.batch_size = 1
        self.epochs = 2

    def evaluate(self):
        evaluation_results = self.trainer.evaluate()
        print(evaluation_results)
    
    def train(self):
        self.trainer.train()

    def init_trainer(self, data, tokenizer, lora=False):
        model = self.get_lora_model(data) if lora else self.get_model(data)

        self.trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./data/sentiment_analysis",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            ),
            train_dataset=data.get_tokenized_train_dataset(),
            eval_dataset=data.get_tokenized_test_dataset(),
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=self.compute_metrics,
        )

    def get_model(self, data):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=data.get_id2label(),
            label2id=data.get_label2id(),
        )

        for param in model.base_model.parameters():
            param.requires_grad = True

        return model

    def get_lora_model(self, data):
        model = self.get_model(data)
        config = LoraConfig(r=8, lora_alpha=16)
        lora_model = get_peft_model(model, config)
        return lora_model

    def compute_metrics(self, eval_pred):
        # Compute loss
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        preds_loss = np.argmax(preds, axis=1)
        eval_loss = ((preds_loss - eval_pred.label_ids) ** 2).mean().item()

        # Compute accuracy
        preds_acc = np.argmax(eval_pred.predictions, axis=1)
        accuracy = (preds_acc == eval_pred.label_ids).mean()

        return {"accuracy": accuracy, "eval_loss": eval_loss}

    def train_with_own_loop(self, data, lora=False):
        model = self.get_lora_model(data) if lora else self.get_model(data)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        loss_function = CrossEntropyLoss()
        num_classes = 2
        train_data_loader = DataLoader(data.get_tokenized_train_dataset(), batch_size=self.batch_size, shuffle=True)
        val_data_loader = DataLoader(data.get_tokenized_test_dataset(), batch_size=self.batch_size, shuffle=False)
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            num_batches = len(train_data_loader)
            for i, batch in enumerate(train_data_loader):
                input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
                labels = torch.tensor(batch['labels']).unsqueeze(0)

                # Forward pass
                outputs = model(input_ids=input_ids)
                loss = loss_function(outputs.logits.view(-1, num_classes), labels.view(-1))
                print("Epoch:", epoch, "Batch:", i, "of", num_batches, "| Loss:", loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluation
            model.eval()
            total_eval_loss = 0
            total_eval_accuracy = 0
            for batch in val_data_loader:
                input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
                labels = torch.tensor(batch['labels']).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    loss = loss_function(outputs.logits.view(-1, num_classes), labels.view(-1))
                    total_eval_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)
                    total_eval_accuracy += (preds == labels).float().mean().item()

            avg_val_loss = total_eval_loss / len(val_data_loader)
            avg_val_accuracy = total_eval_accuracy / len(val_data_loader)
            print(f"Validation Loss: {avg_val_loss}")
            print(f"Validation Accuracy: {avg_val_accuracy}")

if __name__ == "__main__":
    data = imdb().reduce_to_fraction(0.01)
    tokenizer_wrapper = TokenizerWrapper("gpt2")
    trainer_wrapper = TrainerWrapper("gpt2")
    
    lora_model = trainer_wrapper.get_lora_model(data)
    print(lora_model)

    #trainer_wrapper.init_trainer(data, tokenizer_wrapper.get_tokenizer())
    #trainer_wrapper.train()
    #trainer_wrapper.evaluate()


''' The alternative way of training:

    
'''

# For later:

#lora_model.save_pretrained("mwolfram/gpt2-lora")

# from peft import AutoPeftModelForCausalLM
# from transformers import OPTForCausalLM

# #lora_model = AutoPeftModelForCausalLM.from_pretrained("mwolfram/opt-350m-lora")

# lora_model = OPTForCausalLM.from_pretrained("mwolfram/opt-350m-lora")

# input_ids = tokenizer.encode('Hello, I am a', return_tensors='pt')
# generated_text = lora_model.generate(input_ids, max_length=50)
# print(tokenizer.decode(generated_text[0]))