import torch
from torch.utils.data import DataLoader
import json
import numpy as np
from transformers import BertTokenizer
from sklearn.metrics import f1_score

from ..utils.data_loading import preprocess_text
from ..utils.dataset import Subtask1Dataset
from .train_each_fold import Subtask1Model


def load_test_data_with_labels(test_data_path):
    # Load test data with labels
    with open(test_data_path, 'r', encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file.readlines()]
    
    # Extract texts and labels
    texts = [preprocess_text(item['text']) for item in test_data]
    labels = []
    for item in test_data:
        label = item['labels']
        labels.append([label['bin_maj'], label['bin_one'], label['bin_all'], label['multi_maj'], label['disagree_bin']])

    return texts, labels

def ensemble_predict(models, dataloader, device):
    all_preds = []
    
    for model in models:
        model.eval()
        preds = {task: [] for task in ["bin_maj", "bin_one", "bin_all", "multi_maj", "disagree_bin"]}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                if isinstance(logits, torch.Tensor):
                    logits = (logits,)

                for i, task in enumerate(preds.keys()):
                    if task in ["bin_maj", "bin_one", "bin_all", "disagree_bin"]:
                        task_preds = torch.sigmoid(logits[i]).cpu().numpy()
                    elif task == "multi_maj":
                        task_preds = torch.nn.functional.softmax(logits[i], dim=1).cpu().numpy()
                    preds[task].extend(task_preds)

        all_preds.append(preds)
    
    # Aggregate predictions
    final_preds = {task: [] for task in preds.keys()}
    for task in final_preds.keys():
        task_preds = np.mean([np.array(model_preds[task]) for model_preds in all_preds], axis=0)
        if task == "multi_maj":
            final_preds[task] = np.argmax(task_preds, axis=1)
        else:
            final_preds[task] = np.round(task_preds).astype(int)
    
    return final_preds

def evaluate_ensemble_on_test_set(models, test_data_path, tokenizer_path, device):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load test data with labels
    test_texts, test_labels = load_test_data_with_labels(test_data_path)
    test_dataset = Subtask1Dataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Perform ensemble prediction
    preds = ensemble_predict(models, test_loader, device)
    
    # Prepare true labels
    targets = {task: [label[i] for label in test_labels] for i, task in enumerate(["bin_maj", "bin_one", "bin_all", "multi_maj", "disagree_bin"])}
    
    # Evaluate predictions
    f1_scores = {task: f1_score(targets[task], preds[task], average='macro') for task in preds.keys()}
    avg_f1 = np.mean(list(f1_scores.values()))
    
    print(f'F1 Scores: {f1_scores}')
    print(f'Average F1 Score: {avg_f1}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_path = 'models/pretrained_model'

    models = []
    for fold_idx in range(5):
        model = Subtask1Model(tokenizer_path, hidden_size_1=768, hidden_size_2=256, dropout_rate=0.3606437739294118).to(device)
        model.load_state_dict(torch.load(f'models/competition/best_model_fold_{fold_idx}.pt'))
        models.append(model)

    evaluate_ensemble_on_test_set(models, 'data/germeval/germeval-development-test-with-labels.jsonl', tokenizer_path, device)

if __name__ == '__main__':
    main()
