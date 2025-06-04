import string

def clean_and_tokenize(text, to_lowercase=True, remove_punct=True):
    if to_lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
