import re

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Replace slash with space (prevents "tensorflowpytorch" concatenation)
        text = text.replace('/', ' ')
        # Keep alphanumeric, spaces, dash, dot, plus, hash
        text = re.sub(r'[^a-zA-Z0-9\s\-\.\+#]', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> list:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def extract_words(text: str) -> list:
        words = text.split()
        return [w for w in words if len(w) > 2]