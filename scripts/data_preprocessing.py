import re
import numpy as np
from keras.preprocessing import text, sequence

def clean_text(sentences):
    """Clean sentences by removing URLs and digits."""
    for i in range(len(sentences)):
        sentences[i] = re.sub('\d','0', sentences[i])
        sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", sentences[i])
    return sentences

def preprocess_data(train_sentences, test_sentences, max_features=20000, maxlen=100):
    """Tokenizes, sequences, and pads the sentences."""
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_sentences)
    
    tokenized_train = tokenizer.texts_to_sequences(train_sentences)
    X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    
    tokenized_test = tokenizer.texts_to_sequences(test_sentences)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
    
    return X_train, X_test, tokenizer.word_index
