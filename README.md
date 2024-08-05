# GenAI-lightweight-fine-tuning
Apply Lightweight Fine-Tuning to a Foundation Model (Udacity Generative AI Nanodegree)

Use `LightweightFineTuning.py` to run the project.

## Project Rubric and Solutions

Submission Requirements and how they are solved.

### Load a pretrained HF model

> Includes the relevant imports and loads a pretrained Hugging Face model that can be used for sequence classification

The model is loaded in `TrainerWrapper.py`, using the `init_trainer` method. It will in turn use `get_model` or `get_lora_model` to get the model for training and evaluation.

### Load and preprocess a dataset

> Includes the relevant imports and loads a Hugging Face dataset that can be used for sequence classification. Then includes relevant imports and loads a Hugging Face tokenizer that can be used to prepare the dataset. A subset of the full dataset may be used to reduce computational resources needed.

`DatasetWrapper.py` can load both `imdb` and `sms_spam` datasets, the chosen dataset is loaded in `load_dataset`. The amount of data can be downsized to some fraction, and the `train`and `test`sets can be queried separately.

I chose to go forward with the `sms_spam` dataset.

The dataset can be preprocessed using the `TokenizerWrapper.py`. This is shown in `LightweightFineTuning.py`, by simply calling `tokenizer_wrapper.tokenize(data)`. This will take care of tokenizing all data splits. The `DatasetWrapper` takes care of storing the tokenization results.

### Evaluate the pretrained model

> At least one classification metric is calculated using the dataset and pretrained model

I am using the HuggingFace Trainer for evaluation. This is invoked in the `TrainerWrapper.py` by first initializing a trainer with `init_trainer` and then calling `evaluate` on the wrapper.

I chose to go ahead with the `facebook/opt-350` model.
When I evaluated the model with the untrained classification head, I got these results:

```
{'eval_loss': 1.6337233781814575, 'eval_model_preparation_time': 0.0026, 'eval_accuracy': 0.2116591928251121, 'eval_runtime': 25.4441, 'eval_samples_per_second': 43.822, 'eval_steps_per_second': 43.822}
```

I also evaluated it with the self-written evaluation process in `TrainerWrapper::evaluate_with_own_loop` and got the same results:

```
Validation Loss: 1.6337234101188265
Validation Accuracy: 0.2116591928251121
```

I couldn't even train the full model, as my GPU memory would be insufficient for that.

### Create a PEFT model

> Includes the relevant imports, initializes a Hugging Face PEFT config, and creates a PEFT model using that config

I chose to create a LoraModel, using This happens in `TrainerWrapper::get_lora_model`. The model is based on `facebook/opt-350` as a Foundation Model. `rate` is set to 8, `lora_alpha` is set to 16. These are fairly standard values and yielded good results.

### Train the PEFT model

> The model is trained for at least one epoch using the PEFT model and dataset

As opposed to the original model, I chose to train the PEFT Model with my own training loop instead of HuggingFace's trainer, as the HF trainer had trouble training the Lora Model. This can be seen in `TrainerWrapper::train_with_own_loop`.

The Lora version of `facebook/opt-350` worked insanely fast in training, I also had no trouble with my GPU memory here. I could comfortably train two epochs on the full `sms_spam` dataset.

I ended up getting the following results:

Lora version of `facebook/opt-350`
```
Validation Loss: 0.024231021698545237
Validation Accuracy: 0.9937219730941704
```

Given the efficient training phase, this is a really good result. I compared this to training the full distilbert-base-uncased in a similar way, which took longer and gave me a slightly worse performance:

Comparison with full `distilbert-base-uncased`
```
Validation Loss: 0.04222637224472589
Validation Accuracy: 0.9819819819819819
```

### Save the PEFT model

> Fine-tuned parameters are saved to a separate directory. The saved weights directory should be in the same home directory as the notebook file.

This is done at the end of `TrainerWrapper::train_with_own_loop`, so right after training.

### Load the saved PEFT model

> Includes the relevant imports then loads the saved PEFT model

Called as `model = trainer_wrapper.load_peft_model(model_path)`

### Evaluate the fine-tuned model

> Repeats the earlier evaluation process (same metric(s) and dataset) to compare the fine-tuned version to the original version of the model

Called as `trainer_wrapper.evaluate_with_own_loop(data, model)`
