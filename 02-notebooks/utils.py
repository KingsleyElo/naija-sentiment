import re
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def dataset_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


import re
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def dataset_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


def preparation_for_model_train(X_train, X_val, X_test,
                                y_train, y_val, y_test,
                                vocab_size=50000, max_len=50,
                                tokenizer=None):
    # this is to avoid recreating and refiting a new tokenizer
    
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(X_train)

    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val),   maxlen=max_len)
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_len)

    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    y_train = y_train.map(label_mapping).values
    y_val   = y_val.map(label_mapping).values
    y_test  = y_test.map(label_mapping).values

    return X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, tokenizer


def load_tokenizer(path='../03-models/rnn_tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)