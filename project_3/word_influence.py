import os
import pandas as pd
import re
import sys
import importlib
import plotly.graph_objects as go



from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


import testing_main


tfidf_vectorizer = TfidfVectorizer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

# Disable the specific UserWarning
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

######################################################################################################

current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
src_data_directory = os.path.join(current_directory, "src", "data")
sys.path.append(src_data_directory)
import data_cleaner
importlib.reload(data_cleaner)
# File name is data_cleaner

######################################################################################################

def df_splitter_and_vectorizer(df):                    
            """
                Function for splitting and vectorizing data.
                returns split and vectorized data. 

            """
            X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], 
                                                    df[['answer_type','metadata_language', 'metadata_skills', 'metadata_task', 
                                                        'metadata_category', 'metadata_context', 'metadata_grade',
                                                        'metadata_split', 'metadata_source', 'metadata_img_height', 
                                                        'metadata_img_width']], 
                                                    test_size=0.4, random_state=42)
            
            

            # Fit the vectorizer on the preprocessed text data and transform it
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            return X_train_tfidf, X_test_tfidf, y_train, y_test

######################################################################################################


def calculate_word_influence(sentence, svm_model, tfidf_vectorizer):

    """

        Calculates the influence of each word on the prediction made. 
        Uses both SVM coefficients and TF IDF coefficients. 
        Returns the influence of each word. 
    
    """

    
    scale_tfidf=1.0 
    scale_coeff=1.0

    # Preprocess the input sentence
    sentence = sentence.lower()
    words = sentence.split()

    # Transform the preprocessed sentence into a TF-IDF feature vector
    sentence_vector = tfidf_vectorizer.transform([sentence])

    # Get the coefficients from the SVM model
    coefficients = svm_model.coef_.toarray()[0]

    # Get the feature names from the TF-IDF vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()  # Convert to list

    # Calculate the influence of each word in the sentence
    word_influence = {}
    for word in words:
        # Get the index of the word in the feature names
        try:
            word_index = feature_names.index(word)
        except ValueError:
            # If the word is not found in feature names, set influence to 0
            word_influence[word] = 0
            continue  # Move to the next word

        # Calculate the influence of the word using both TF-IDF value and coefficient
        tfidf_value = sentence_vector[0, word_index]
        coefficient = coefficients[word_index]
        influence = (scale_tfidf * tfidf_value) + (scale_coeff * coefficient)

        # Store the influence of the word
        word_influence[word] = influence

    return word_influence



######################################################################################################

def test_text_processor_for_onlyImage(text, answer_type, X_train_tfidf, y_train, result, word_influence_list, plot_output):
    """
    Makes prediction and gives output, along with influence printed.
    """

    SVM = svm.SVC(C=10, kernel='linear', degree=2, gamma='auto') # Parameters obtained after optimizing
    SVM.fit(X_train_tfidf, y_train[answer_type])    # Fit the SVM classifier on the training dataset

    preprocessed_text = data_cleaner.preprocessing_text(text)   # Preprocess the input text
    preprocessed_text = tfidf_vectorizer.transform([preprocessed_text])    # Transform preprocessed text using tfidf_vectorizer

    prediction = SVM.predict(preprocessed_text)     # Predict the label
    result.append(prediction[0])
    print("Prediction for " + answer_type + ":", prediction)

    word_influence = calculate_word_influence(text, SVM, tfidf_vectorizer)  # Finding the influence of each word
    # print("Influence of each word in the sentence:")
    # for word, influence in word_influence.items():
    #     print(f"{word}: {influence}")

    if plot_output:
        plot_word_influence(word_influence, answer_type, prediction)  # Visualize word influence using Plotly

    word_influence_list.append(word_influence)

    return result, word_influence_list


######################################################################################################

def count_words(sentence):
    # Split the sentence into words using whitespace as delimiter
    words = sentence.split()
    # Count the number of words
    num_words = len(words)
    return num_words



######################################################################################################

def predictor_for_Question( Question, X_train_tfidf, y_train, plot_output):

    """
        Makes Prediction for each category for a given prompt.
        Appends results in a list. 
    """

    result = []
    word_influence_list = []
    # Looping through all classes
    for answer_type in ['answer_type','metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language']:
        result,word_influence_list = test_text_processor_for_onlyImage(Question, answer_type, X_train_tfidf, y_train, result, word_influence_list,plot_output )

        print ('\n')

    return word_influence_list
    

######################################################################################################

csv_file_paths = testing_main.get_df_paths()


######################################################################################################

def question_image_influence(question, image_URL):
     
    captions = testing_main.BLIP_Generate_Caption_from_URL(image_URL)

    question = question + " " + captions

    """
        Making Predictions. 
    """

    df = pd.read_csv(csv_file_paths[2])
    X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
    # Question = input("Enter Quesion: ")       
    # print ("\n")
    word_influence = predictor_for_Question( question, X_train_tfidf, y_train , False)

    return word_influence

     


######################################################################################################

    

def class_predictor(Question):

    """
        Making Predictions. 
    """

    df = pd.read_csv(csv_file_paths[0])
    X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
    # Question = input("Enter Quesion: ")       
    # print ("\n")
    predictor_for_Question( Question, X_train_tfidf, y_train, plot_output=True)


######################################################################################################  


def plot_word_influence(word_influence, answer_type, prediction):
    words = list(word_influence.keys())[::-1]  # Reverse the order of words
    influence_values = list(word_influence.values())[::-1]
 
    fig = go.Figure()

    for word, influence in zip(words, influence_values):
        color = 'rgba(50, 171, 96, 0.6)' if influence >= 0 else 'rgba(219, 64, 82, 0.6)'  # Green for non-negative, red for negative
        fig.add_trace(go.Bar(
            y=[word],
            x=[influence],
            orientation='h',
            name=word,
            marker=dict(
                color=color,
                line=dict(
                    color='rgba(50, 171, 96, 1.0)',
                    width=1),
            ),
        ))

    fig.update_layout(
        title=f'Word Influence<br>Answer Type: {answer_type}<br>Prediction: {prediction}',
        yaxis=dict(
            showgrid=True,
            showline=False,
            showticklabels=True,
            domain=[0, 0.85],
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            domain=[0, 0.42],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=100, r=20, t=70, b=70),
    )

    fig.show()


######################################################################################################



# csv_file_paths = testing_main.get_df_paths()    # Getting Data Frame

class_predictor(" WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE  " )


######################################################################################################

#       Testing Data


# QUESTION : WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE 
# QUESTION : WHAT IS THE AREA OF THE FIGURE SHOWN BELOW ? 


# HORSE PULLING A CAR : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIkJDpcdIMFHRjsgvc2JD0RspkUStSXuoIBQ&usqp=CAU
# CAT GIVING HIGH FIVE TO STAUTE OF LIBERTY : https://randomwordgenerator.com/img/picture-generator/52e1d5424b56aa14f1dc8460962e33791c3ad6e04e50744074267bd69149c7_640.jpg
# DIES IMAGE : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzqqxGuOfwNRD61afNt4iX0eBmvcZfCWx5Tg&usqp=CAU
# FOOD CHAIN : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO-8eWbdZ5aEbz1PJ1ryIFYuuG18u8tCc0YXxg33av7SatC5Er8TK9PplEa6IdXsTbQFQ&usqp=CAU


# MAP QUESTION : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSY3NVSNmEeGeEcwByTywefdCi2fncYFkgPcA&usqp=CAU
# PUZZLE PROBLEM : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpcIq0dBVcbISQ9c9lIcqFtd7OnUy9U5reFg&usqp=CAU  # Slight miss classification because of word cones.

#######################################################################