import re

def simple_tokenize(text):
    """Tokenization by lowercasing, stripping non-alphanumeric characters, 
    and splitting text into tokens."""
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return text.split()