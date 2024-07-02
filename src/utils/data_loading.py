import json
import random
from ..utils.preprocessing import preprocess_text

def load_subtask1_data(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    texts = []
    labels = []
    for line in lines:
        data = json.loads(line)
        text = preprocess_text(data['text'])
        annotations = [annotation['label'] for annotation in data['annotations']]
        
        # Count occurrences of each label
        annotation_counts = {}
        for annotation in annotations:
            label = int(annotation.split('-')[0])
            annotation_counts[label] = annotation_counts.get(label, 0) + 1

        total_annotations = len(annotations)
        non_zero_labels = sum(count for lbl, count in annotation_counts.items() if lbl > 0)

        # bin_maj
        bin_maj = 1 if non_zero_labels > total_annotations / 2 else 0
        # bin_one
        bin_one = 1 if non_zero_labels > 0 else 0
        # bin_all
        bin_all = 1 if non_zero_labels == total_annotations and total_annotations > 0 else 0
        # multi_maj
        max_votes = max(annotation_counts.values()) if annotation_counts else 0
        max_labels = [lbl for lbl, count in annotation_counts.items() if count == max_votes]
        multi_maj = random.choice(max_labels) if len(max_labels) > 1 else max_labels[0] if max_labels else 0
        # disagree_bin
        disagree_bin = 1 if 0 in annotation_counts and non_zero_labels > 0 else 0

        texts.append(text)
        labels.append([bin_maj, bin_one, bin_all, multi_maj, disagree_bin])

    return texts, labels
