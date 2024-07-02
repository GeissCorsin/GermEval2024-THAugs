import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn

from ..utils.data_loading import preprocess_text
from ..utils.dataset import Subtask1Dataset
from .train_each_fold import Subtask1Model

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

def load_unlabeled_data(test_data_path):
    with open(test_data_path, 'r', encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file.readlines()]

    texts = [preprocess_text(item['text']) for item in test_data]
    ids = [item['id'] for item in test_data]

    return texts, ids

def create_submission_file(predictions, submission_path, ids):
    multi_maj_mapping = {
        0: '0-Kein',
        1: '1-Gering',
        2: '2-Vorhanden',
        3: '3-Stark',
        4: '4-Extrem'
    }

    with open(submission_path, 'w') as f:
        f.write('id\tbin_maj\tbin_one\tbin_all\tmulti_maj\tdisagree_bin\n')
        for idx, id in enumerate(ids):
            bin_maj = int(predictions["bin_maj"][idx].item())
            bin_one = int(predictions["bin_one"][idx].item())
            bin_all = int(predictions["bin_all"][idx].item())
            multi_maj = multi_maj_mapping[int(predictions["multi_maj"][idx].item())]
            disagree_bin = int(predictions["disagree_bin"][idx].item())
            f.write(f'{id}\t{bin_maj}\t{bin_one}\t{bin_all}\t{multi_maj}\t{disagree_bin}\n')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_texts, test_ids = load_unlabeled_data('data/germeval/germeval-competition-test.jsonl')
    tokenizer = BertTokenizer.from_pretrained('models/pretrained_model')
    test_dataset = Subtask1Dataset(test_texts, [[0, 0, 0, 0, 0]] * len(test_texts), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    models = []
    for fold_idx in range(5):
        model = Subtask1Model('models/pretrained_model', hidden_size_1=768, hidden_size_2=256, dropout_rate=0.3606437739294118).to(device)
        model.load_state_dict(torch.load(f'models/competition/best_model_fold_{fold_idx}.pt'))
        models.append(model)

    predictions = ensemble_predict(models, test_loader, device)
    create_submission_file(predictions, 'submission.tsv', test_ids)
    os.system('zip submission.zip submission.tsv')

if __name__ == '__main__':
    main()
