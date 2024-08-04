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

* **TODO** compare some models
* While I evaluated multiple models, I chose to go forward with `gpt2`.

### Create a PEFT model

> Includes the relevant imports, initializes a Hugging Face PEFT config, and creates a PEFT model using that config

I chose to create a LoraModel, using This happens in `TrainerWrapper::get_lora_model`. The model is based on `gpt2` as a Foundation Model

### Train the PEFT model

> The model is trained for at least one epoch using the PEFT model and dataset

As opposed to the original model, I chose to train the PEFT Model with my own training loop instead of HuggingFace's trainer, as the HF trainer had trouble training the Lora Model.

### Save the PEFT model

> Fine-tuned parameters are saved to a separate directory. The saved weights directory should be in the same home directory as the notebook file.

### Load the saved PEFT model

> Includes the relevant imports then loads the saved PEFT model

### Evaluate the fine-tuned model

> Repeats the earlier evaluation process (same metric(s) and dataset) to compare the fine-tuned version to the original version of the model