import numpy as np
import torch
import pandas as pd
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, EvalPrediction
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
import evaluate
from ..utils.preprocessing import preprocess_text

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Convert labels to int
def convert_labels_to_int(dataset):
    dataset = dataset.map(lambda examples: {'labels': int(examples['labels'])})
    return dataset

# Load the GAHD dataset
df_gahd = pd.read_csv('data/gahd/filtered_gahd_disaggregated.csv')

# Preprocess the dataset
df_gahd = df_gahd.rename(columns={"text": "text", "label": "labels"})
df_gahd['text'] = df_gahd['text'].apply(preprocess_text)
dataset_gahd = Dataset.from_pandas(df_gahd)

# Split the dataset
train_gahd, test_gahd = train_test_split(df_gahd, test_size=0.2, random_state=2024, shuffle=True, stratify=df_gahd['labels'])

# Convert to Hugging Face dataset
train_gahd = Dataset.from_pandas(train_gahd)
test_gahd = Dataset.from_pandas(test_gahd)

# Tokenize and convert labels
train_gahd = train_gahd.map(tokenize_function, batched=True)
test_gahd = test_gahd.map(tokenize_function, batched=True)
train_gahd = convert_labels_to_int(train_gahd)
test_gahd = convert_labels_to_int(test_gahd)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_gahd['labels']), y=train_gahd['labels'])
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define model and trainer
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=2)

# Define metrics using the evaluate library
accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
precision_metric = evaluate.load("precision", trust_remote_code=True)
recall_metric = evaluate.load("recall", trust_remote_code=True)
f1_metric = evaluate.load("f1", trust_remote_code=True)

# Compute metrics function
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_gahd',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_gahd',
    logging_steps=10,
    eval_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_gahd,
    eval_dataset=test_gahd,
    compute_metrics=compute_metrics,
)

# Train the model on GAHD dataset
trainer.train()

# Save the final model
model.save_pretrained('models/finetuned_model_gahd')
tokenizer.save_pretrained('models/finetuned_model_gahd')

print("Final model saved and ready for further fine-tuning on the competition dataset.")
