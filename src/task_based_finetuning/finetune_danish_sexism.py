import re
import emoji
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from ..utils.preprocessing import preprocess_text


# Load the dataset
data = pd.read_csv(r'data\bajer_danish_misogyny\bajer_data_final_20201221.tsv', delimiter='\t')

# Handle NaN values by filling them with appropriate default values
data['text'] = data['text'].fillna('')
data['subtask_A'] = data['subtask_A'].fillna('NOT')
data['subtask_C1'] = data['subtask_C1'].fillna('NaN')  # Use 'NaN' to explicitly handle missing values

# Preprocess the texts
data['text'] = data['text'].apply(preprocess_text)

# Filter for abusive content
data['label_A'] = data['subtask_A'].apply(lambda x: 1 if x == 'ABUS' else 0)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Split the data into training, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data['text'].tolist(), data['label_A'].tolist(), test_size=0.3, random_state=42, stratify=data['label_A'])

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Tokenize the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class MisogynyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = MisogynyDataset(train_encodings, train_labels)
val_dataset = MisogynyDataset(val_encodings, val_labels)
test_dataset = MisogynyDataset(test_encodings, test_labels)

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Load the model with added dropout
config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)

# Define training arguments for very light fine-tuning
training_args = TrainingArguments(
    output_dir='./fine_tuning_results',
    eval_strategy='epoch',  
    save_strategy='epoch',  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  
    weight_decay=0.01,
    logging_dir='./fine_tuning_logs',
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# Define the trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='weighted'),
        'precision': precision_score(p.label_ids, np.argmax(p.predictions, axis=1), average='weighted'),
        'recall': recall_score(p.label_ids, np.argmax(p.predictions, axis=1), average='weighted')
    },
)


trainer.train()


model.save_pretrained('models/fine_tuned-bert')
tokenizer.save_pretrained('models/fine_tuned-bert')

# Evaluate the model on the test set
test_results = trainer.predict(test_dataset)
test_metrics = {
    'accuracy': accuracy_score(test_results.label_ids, np.argmax(test_results.predictions, axis=1)),
    'f1': f1_score(test_results.label_ids, np.argmax(test_results.predictions, axis=1), average='weighted'),
    'precision': precision_score(test_results.label_ids, np.argmax(test_results.predictions, axis=1), average='weighted'),
    'recall': recall_score(test_results.label_ids, np.argmax(test_results.predictions, axis=1), average='weighted')
}

print("Test set metrics:", test_metrics)
