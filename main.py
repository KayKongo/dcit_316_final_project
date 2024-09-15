import bz2
from scripts.data_preprocessing import clean_text, preprocess_data
from scripts.model_training import cudnnlstm_model, train_model
from scripts.model_evaluation import evaluate_model
import numpy as np
import gc

# Load Data
def load_data(train_path, test_path):
    train_file = bz2.BZ2File(train_path)
    test_file = bz2.BZ2File(test_path)
    train_file_lines = [x.decode('utf-8') for x in train_file.readlines()]
    test_file_lines = [x.decode('utf-8') for x in test_file.readlines()]
    
    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]
    test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
    test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]
    
    return train_sentences, test_sentences, train_labels, test_labels

# Main Function
if __name__ == "__main__":
    train_path = '../input/amazonreviews/train.ft.txt.bz2'
    test_path = '../input/amazonreviews/test.ft.txt.bz2'
    
    train_sentences, test_sentences, train_labels, test_labels = load_data(train_path, test_path)
    
    train_sentences = clean_text(train_sentences)
    test_sentences = clean_text(test_sentences)
    
    X_train, X_test, word_index = preprocess_data(train_sentences, test_sentences)
    
    # Define and train the model
    max_features = 20000
    embed_size = 100
    maxlen = 100
    
    embedding_matrix = np.random.normal(size=(min(max_features, len(word_index)), embed_size))
    
    model = cudnnlstm_model(maxlen, max_features, embed_size, embedding_matrix)
    
    model, history = train_model(X_train, train_labels, X_test, test_labels, model)
    
    evaluate_model(model, X_test, test_labels)
    gc.collect()
