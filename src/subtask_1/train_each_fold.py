import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from ..utils.dataset import Subtask1Dataset
from ..utils.evaluation import evaluate_model

class Subtask1Model(nn.Module):
    def __init__(self, bert_model_name, hidden_size_1, hidden_size_2, dropout_rate):
        super(Subtask1Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.bin_maj_classifier = nn.Linear(hidden_size_2, 1)
        self.bin_one_classifier = nn.Linear(hidden_size_2, 1)
        self.bin_all_classifier = nn.Linear(hidden_size_2, 1)
        self.multi_maj_classifier = nn.Linear(hidden_size_2, 5)
        self.disagree_bin_classifier = nn.Linear(hidden_size_2, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        x = self.fc1(pooled_output)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        bin_maj_logits = self.bin_maj_classifier(x)
        bin_one_logits = self.bin_one_classifier(x)
        bin_all_logits = self.bin_all_classifier(x)
        multi_maj_logits = self.multi_maj_classifier(x)
        disagree_bin_logits = self.disagree_bin_classifier(x)
        return bin_maj_logits, bin_one_logits, bin_all_logits, multi_maj_logits, disagree_bin_logits, pooled_output

def compute_class_weights(labels):
    multi_labels = [label[3] for label in labels]
    class_sample_count = np.bincount(multi_labels)
    class_weights = 1. / class_sample_count
    return torch.tensor(class_weights, dtype=torch.float)

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_fold(fold_idx, device):
    with open(f'data/germeval/folds/fold_{fold_idx}.json', 'r') as f:
        fold_data = json.load(f)
    
    tokenizer = BertTokenizer.from_pretrained('models/pretrained_model')
    train_dataset = Subtask1Dataset(fold_data['train_texts'], fold_data['train_labels'], tokenizer)
    val_dataset = Subtask1Dataset(fold_data['val_texts'], fold_data['val_labels'], tokenizer)

    model = Subtask1Model('models/pretrained_model', hidden_size_1=768, hidden_size_2=256, dropout_rate=0.3606437739294118).to(device)
    class_weights = compute_class_weights(fold_data['train_labels']).to(device)
    
    train_and_evaluate_subtask1(train_dataset, val_dataset, model, device, class_weights, fold_idx)

    print(f"Fold {fold_idx} trained and saved.")

def train_and_evaluate_subtask1(train_dataset, val_dataset, model, device, class_weights, fold_idx):
    params = {
        'lr': 1e-05,
        'batch_size': 16,
        'weight_decay': 0.20190110569667427,
        'epochs': 12
    }
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    loss_fct_bin = torch.nn.BCEWithLogitsLoss()
    loss_fct_multi = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    best_avg_f1 = 0
    patience = 4
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            bin_maj_logits, bin_one_logits, bin_all_logits, multi_maj_logits, disagree_bin_logits, _ = model(input_ids, attention_mask)
            bin_labels = labels[:, :3]
            multi_labels = labels[:, 3].long()
            disagree_bin_labels = labels[:, 4]
            bin_maj_loss = loss_fct_bin(bin_maj_logits.view(-1), bin_labels[:, 0].view(-1))
            bin_one_loss = loss_fct_bin(bin_one_logits.view(-1), bin_labels[:, 1].view(-1))
            bin_all_loss = loss_fct_bin(bin_all_logits.view(-1), bin_labels[:, 2].view(-1))
            multi_maj_loss = loss_fct_multi(multi_maj_logits, multi_labels)
            disagree_bin_loss = loss_fct_bin(disagree_bin_logits.view(-1), disagree_bin_labels.view(-1))
            loss = bin_maj_loss + bin_one_loss + bin_all_loss + multi_maj_loss + disagree_bin_loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)

        f1_scores, avg_f1, _, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step(avg_f1)
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, F1 Scores: {f1_scores}')
        print(f'Validation Average F1 Score: {avg_f1}')

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            patience_counter = 0
            ensure_dir_exists('models/competition')
            torch.save(model.state_dict(), f'models/competition/best_model_fold_{fold_idx}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold_idx in range(5):
        if not os.path.exists(f'models/competition/best_model_fold_{fold_idx}.pt'):
            train_fold(fold_idx, device)

if __name__ == '__main__':
    main()
