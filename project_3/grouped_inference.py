import os
import sys
import numpy as np
import pandas as pd
from sklearn import svm 
import testing_main
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
import importlib
from sklearn.model_selection import train_test_split
import word_influence
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import seaborn as sns

import itertools

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10




current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
src_data_directory = os.path.join(current_directory, "src", "data")
sys.path.append(src_data_directory)
import data_cleaner
importlib.reload(data_cleaner)
# File name is data_cleaner

def df_splitter_and_vectorizer(df):                    
            """
                Function for splitting and vectorizing data.
                returns split and vectorized data. 

            """
            X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], 
                                               df['metadata_task'],  # Only use 'metadata_task' column
                                               test_size=0.4, random_state=42)
            
            

            # Fit the vectorizer on the preprocessed text data and transform it
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            return X_train_tfidf, X_test_tfidf, y_train, y_test

def calculate_sentence_vector(sentence, svm_model, tfidf_vectorizer):
    """
    Calculates the sentence vector by applying weighted averaging of word vectors.
    """

    scale_tfidf = 1.0 
    scale_coeff = 1.0

    # Preprocess the input sentence
    sentence = data_cleaner.preprocessing_text(sentence)
    words = sentence.split()

    # Transform the preprocessed sentence into a TF-IDF feature vector
    sentence_vector = tfidf_vectorizer.transform([sentence])

    # Get the coefficients from the SVM model
    coefficients = svm_model.coef_.toarray()[0]

    # Get the feature names from the TF-IDF vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()  # Convert to list

    # Initialize variables for weighted averaging
    weighted_word_vectors = []
    total_weight = 0

    # Calculate the influence of each word in the sentence and compute weighted average
    for word in words:
        # Get the index of the word in the feature names
        try:
            word_index = feature_names.index(word)
        except ValueError:
            # If the word is not found in feature names, continue to the next word
            continue

        # Calculate the influence of the word using both TF-IDF value and coefficient
        tfidf_value = sentence_vector[0, word_index]
        coefficient = coefficients[word_index]
        influence = (scale_tfidf * tfidf_value) + (scale_coeff * coefficient)

        # Compute weighted word vector
        word_vector = tfidf_vectorizer.transform([word]).toarray()[0]  # Assuming tfidf_vectorizer is fitted
        weighted_word_vector = influence * word_vector

        # Accumulate weighted word vectors and their total weight
        weighted_word_vectors.append(weighted_word_vector)
        total_weight += influence

    # Compute weighted average of word vectors
    if total_weight > 0:
        sentence_vector = np.sum(weighted_word_vectors, axis=0) / total_weight
    else:
        # If no word has influence, return zero vector
        sentence_vector = np.zeros_like(weighted_word_vectors[0])

    return sentence_vector

def train_svm_model(df):
    """
    Trains an SVM model on the entire dataset.
    
    Args:
    - df (DataFrame): DataFrame containing the dataset.
    
    Returns:
    - svm_model: Trained SVM model.
    """
    # Preprocess data and vectorize
    X_train_tfidf, _, y_train, _ = df_splitter_and_vectorizer(df)

    # Train SVM model on the entire dataset
    svm_model = svm.SVC(C=10, kernel='linear', degree=2, gamma='auto') # Parameters obtained after optimizing
    svm_model.fit(X_train_tfidf, y_train)    # Fit the SVM classifier on the entire dataset

    return svm_model


def grouped_influence_finder(text, answer_type, X_train_tfidf, y_train, result, word_influence_list, plot_output, svm_model, tfidf_vectorizer):
    """
    Makes prediction and gives output, along with influence printed.
    """

    preprocessed_text = data_cleaner.preprocessing_text(text)   # Preprocess the input text
    preprocessed_text = tfidf_vectorizer.transform([preprocessed_text])    # Transform preprocessed text using tfidf_vectorizer

    prediction = svm_model.predict(preprocessed_text)     # Predict the label
    result.append(prediction[0])
    print("Prediction for " + answer_type + ":", prediction)

    word_influence_1 = word_influence.calculate_word_influence(text, svm_model, tfidf_vectorizer)  # Finding the influence of each word
    word_influence_list.append(word_influence_1)

    # Calculate sentence vector
    sentence_vector = calculate_sentence_vector(text, svm_model, tfidf_vectorizer)

    return result, word_influence_list, sentence_vector



def get_sentence_vectors(Question, svm_model, tfidf_vectorizer):
    """
    Calculates sentence vectors for the input question.
    
    Args:
    - Question (str): Input question.
    - svm_model: Trained SVM model.
    - tfidf_vectorizer: TF-IDF vectorizer.
    
    Returns:
    - result: Predicted label for the question.
    - word_influence_list: List of word influences.
    - sentence_vectors: Sentence vectors.
    """
    result = []
    word_influence_list = []

    # Vectorize the input question using the TF-IDF vectorizer
    question_tfidf = tfidf_vectorizer.transform([Question])

    # Make predictions and calculate influences
    prediction, word_influence_list, sentence_vectors = grouped_influence_finder(Question, "metadata_task", question_tfidf, None, result, word_influence_list, plot_output=False, svm_model=svm_model, tfidf_vectorizer=tfidf_vectorizer)

    result.append(prediction[0])
    return result, word_influence_list, sentence_vectors


def visualize_non_zero_elements(sentence_vector):
    non_zero_indices = np.nonzero(sentence_vector)[0]
    non_zero_values = sentence_vector[non_zero_indices]
    
    print("Non-zero indices:", non_zero_indices)
    print("Corresponding values:", non_zero_values)

# visualize_non_zero_elements(sentence_vectors)


def visualize_sentence_vectors_from_user_prompt(sentences, svm_model, tfidf_vectorizer):
    """
    Visualizes the sentence vectors of the given sentences in 2D space after dimensionality reduction.
    
    Args:
    - sentences (list): A list of strings representing the input sentences.
    - svm_model: Trained SVM model.
    - tfidf_vectorizer: TF-IDF vectorizer.
    
    Returns:
    - None
    """
    # Step 1: Obtain sentence vectors for the given sentences
    sentence_vectors = []
    for sentence in sentences:
        _, _, sentence_vector = get_sentence_vectors(sentence, svm_model, tfidf_vectorizer)
        sentence_vectors.append(sentence_vector)
    
    # Step 2: Perform dimensionality reduction to 2D
    pca = PCA(n_components=2)
    sentence_vectors_2d = pca.fit_transform(np.array(sentence_vectors))
    
    # Step 3: Plot the 2D representations
    plt.scatter(sentence_vectors_2d[:, 0], sentence_vectors_2d[:, 1], label='Sentence Vectors')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Visualization of Sentence Vectors')
    plt.legend()
    plt.show()


def visualize_sentence_vectors_from_dataframe_column(data_frame, sentence_column, label_column, perplexity, n_iter, svm_model, tfidf_vectorizer, num_entries=5000):
 
    # Extract sentences and true labels for a subset of entries
    sentences = data_frame[sentence_column].iloc[:num_entries].tolist()
    true_labels = data_frame[label_column].iloc[:num_entries].tolist()
    
    # Obtain sentence vectors for the given sentences
    sentence_vectors = []
    for sentence in sentences:
        _, _, sentence_vector = get_sentence_vectors(sentence, svm_model, tfidf_vectorizer)
        sentence_vectors.append(sentence_vector)
    
    # Perform dimensionality reduction to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    sentence_vectors_2d = tsne.fit_transform(np.array(sentence_vectors))
    
    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    for label in set(true_labels):
        indices = [i for i, l in enumerate(true_labels) if l == label]
        plt.scatter(sentence_vectors_2d[indices, 0], sentence_vectors_2d[indices, 1], label=f'Class {label}', s=100)
    

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Scatter Plot of Sentence Vectors after t-SNE')
    plt.legend()
    plt.show()



def main():
     

    csv_file_paths = testing_main.get_df_paths()
    df = pd.read_csv(csv_file_paths[0])


    svm_model = train_svm_model(df)

    visualize_sentence_vectors_from_dataframe_column(df, 'question', 'metadata_task', perplexity=1000, n_iter=210000, svm_model=svm_model, tfidf_vectorizer=tfidf_vectorizer)

main()