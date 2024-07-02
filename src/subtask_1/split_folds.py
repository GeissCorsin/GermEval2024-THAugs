import os
import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import BertTokenizer

from ..utils.data_loading import load_subtask1_data

def save_folds(texts, labels, k=5):
    # Convert labels to numpy array for stratification
    labels = np.array(labels)
    
    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=2024)
    folds = []
    
    for fold_idx, (train_index, val_index) in enumerate(mskf.split(texts, labels)):
        train_texts = [texts[i] for i in train_index]
        train_labels = [labels[i].tolist() for i in train_index]  # Convert back to list
        val_texts = [texts[i] for i in val_index]
        val_labels = [labels[i].tolist() for i in val_index]  # Convert back to list
        
        fold_data = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels
        }
        
        os.makedirs('data/germeval/folds', exist_ok=True)
        with open(f'data/germeval/folds/fold_{fold_idx}.json', 'w') as f:
            json.dump(fold_data, f)

def main():
    tokenizer = BertTokenizer.from_pretrained('models/pretrained_model')
    texts, labels = load_subtask1_data('data/germeval/germeval-competition-traindev.jsonl', tokenizer)
    
    if not os.path.exists('data/germeval/folds'):
        save_folds(texts, labels)

if __name__ == '__main__':
    main()
