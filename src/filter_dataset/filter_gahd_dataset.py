import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from ..utils.preprocessing import preprocess_text


def read_and_prepare_data(file_path):
    """Read and preprocess data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = preprocess_text(item['text'])
            labels = [annotation['label'] for annotation in item['annotations']]
            label = 1 if any(l != '0-Kein' for l in labels) else 0
            data.append({'text': text, 'label': label})
    return data

# Paths to data and model
data_file_path = 'data/germeval/germeval-competition-traindev.jsonl'
model_path = 'models/finetuned/sentence_transformers/finetuned_model'
dataset_path = 'data/gahd/gahd_disaggregated.csv'
filtered_dataset_path = 'data/gahd/filtered_gahd_disaggregated.csv'

# Load and preprocess the Germeval dataset
data = read_and_prepare_data(data_file_path)
known_sexist_sentences = [example['text'] for example in data if example['label'] == 1]

# Load the fine-tuned model
model = SentenceTransformer(model_path)

# Load and preprocess the gahd dataset
data = pd.read_csv(dataset_path)
data['preprocessed_text'] = data['text'].apply(preprocess_text)

# Compute embeddings
embeddings = model.encode(data['preprocessed_text'].tolist(), convert_to_tensor=True)
known_sexist_embeddings = model.encode(known_sexist_sentences, convert_to_tensor=True)

# Compute similarity
similarities = [util.pytorch_cos_sim(embedding, known_sexist_embeddings).mean().item() for embedding in embeddings]
data['similarity'] = similarities

# Filter dataset based on the threshold for sexist texts only
threshold = 0.14007388055324554
sexist_data = data[(data['label'] == 1) & (data['similarity'] > threshold)]

# Balance the dataset
label_1_count = sexist_data.shape[0]
label_0_data = data[data['label'] == 0]
balanced_label_0_data = label_0_data.sample(n=label_1_count, random_state=2024) if label_0_data.shape[0] >= label_1_count else label_0_data

# Combine and save the balanced dataset
final_data = pd.concat([sexist_data, balanced_label_0_data])
final_data = final_data[['text', 'similarity', 'label', 'annotator_labels', 'expert_labels']]
final_data.to_csv(filtered_dataset_path, index=False)

print(f"Filtered dataset saved to {filtered_dataset_path}")
