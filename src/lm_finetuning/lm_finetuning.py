import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Define a custom dataset class for language modeling
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

# Load and preprocess the data
def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    texts = [json.loads(line)['text'] for line in lines]
    return texts

# Load the GermEval texts
train_file_path = 'data/germeval/germeval-competition-traindev.jsonl'
texts = load_texts(train_file_path)

# Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('deepset/gbert-large')
dataset = TextDataset(texts, tokenizer)

# Initialize the model
model = BertForMaskedLM.from_pretrained('deepset/gbert-large')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('models/pretrained_model')
tokenizer.save_pretrained('models/pretrained_model')
