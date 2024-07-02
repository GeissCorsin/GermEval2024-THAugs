import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, precision_recall_curve
from ..utils.preprocessing import preprocess_text

class PairsDataset(Dataset):
    """Custom Dataset class for pairs."""
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return InputExample(texts=[example.texts[0], example.texts[0]])

def read_and_prepare_data(file_path):
    """Read the JSONL file and prepare data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = preprocess_text(item['text'])
            labels = [annotation['label'] for annotation in item['annotations']]
            label = 1 if any(l != '0-Kein' for l in labels) else 0
            data.append(InputExample(texts=[text], label=label))
    return data

def find_optimal_threshold(similarities, labels, target_precision=0.75):
    """Find the optimal threshold ensuring high precision."""
    precision, recall, thresholds = precision_recall_curve(labels, similarities)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    valid_indices = np.where(precision >= target_precision)[0]
    if len(valid_indices) == 0:
        raise ValueError("No thresholds found that meet the target precision.")
    
    best_index = valid_indices[np.argmax(f1_scores[valid_indices])]
    best_threshold = thresholds[best_index]
    return best_threshold, precision[best_index], recall[best_index], f1_scores[best_index]

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, known_sexist_embeddings, test_data, test_labels, device):
    """Evaluate the model."""
    test_texts = [example.texts[0] for example in test_data]
    test_embeddings = model.encode(test_texts, convert_to_tensor=True).to(device)

    similarities = [util.pytorch_cos_sim(embedding, known_sexist_embeddings).mean().item() for embedding in test_embeddings]

    best_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(similarities, test_labels, target_precision=0.75)

    predictions = [1 if sim > best_threshold else 0 for sim in similarities]

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)

    print(f"Optimal Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    plot_confusion_matrix(test_labels, predictions, classes=['Non-Sexist', 'Sexist'])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file_path = 'data/germeval/germeval-competition-traindev.jsonl'
    data = read_and_prepare_data(data_file_path)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=2024, stratify=[example.label for example in data])

    model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    model = SentenceTransformer(model_name).to(device)

    known_sexist_sentences = [example.texts[0] for example in train_data if example.label == 1]
    known_sexist_embeddings = model.encode(known_sexist_sentences, convert_to_tensor=True).to(device)

    print("Evaluation before fine-tuning:")
    evaluate_model(model, known_sexist_embeddings, test_data, [example.label for example in test_data], device)

    train_examples = [InputExample(texts=[example.texts[0], example.texts[0]]) for example in train_data if example.label == 1]
    train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, output_path='models/finetuned/sentence_transformers/finetuned_model')

    model.save('models/finetuned/sentence_transformers')

    model = SentenceTransformer('models/sentence_transformers').to(device)

    known_sexist_embeddings = model.encode(known_sexist_sentences, convert_to_tensor=True).to(device)

    print("Evaluation after fine-tuning:")
    evaluate_model(model, known_sexist_embeddings, test_data, [example.label for example in test_data], device)

    new_sentence = "Ich hasse Frauen."
    new_embedding = model.encode(new_sentence, convert_to_tensor=True).to(device)
    similarity = util.pytorch_cos_sim(new_embedding, known_sexist_embeddings).mean().item()
    
    print(f"Similarity of the new sentence to known sexist sentences: {similarity}")

if __name__ == '__main__':
    main()
