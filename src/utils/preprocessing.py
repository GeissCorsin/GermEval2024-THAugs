import re
import emoji

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = "".join([c for c in text if c not in emoji.EMOJI_DATA]).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

