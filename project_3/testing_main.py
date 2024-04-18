import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import datasets
import os
import io
import ast
from io import BytesIO
import matplotlib.pyplot as plt

from datasets import load_dataset
import datasets
import pandas as pd
import re
import sys
import importlib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from langdetect import detect

from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


tfidf_vectorizer = TfidfVectorizer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

# Disable the specific UserWarning
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
src_data_directory = os.path.join(parent_directory, "src", "data")
#print ("\n\n")
#print (src_data_directory)
sys.path.append(src_data_directory)
import data_cleaner
importlib.reload(data_cleaner)
# File name is data_cleaner




import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    #print (x)
else:
    print ("MPS device not found.")




processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.to(mps_device)


metadata_task_unique_class = ['figure question answering','visual question answering','math word problem','geometry problem solving','textbook question answering']
metadata_category_unique_class = ['general-vqa','math-targeted-vqa']
answer_type_unique_class = ['integer','text','float','list']
metadata_context_unique_class = ['scatter plot','synthetic scene','table','geometry diagram','bar chart','abstract scene','function plot','line plot','natural image','puzzle test','scientific figure','pie chart','map chart','medical image','document image','radar chart','heatmap chart','word cloud']
metadata_grade_unique_class = ['daily life','elementary school','high school','college']
metadata_language_unique_class = ['chinese','english', 'persian']

#'metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language'


def BLIP_Generate_Caption(image_data):

    raw_image = image_data.convert('RGB')
    raw_image_tensor = torch.tensor(np.array(raw_image)).permute(2, 0, 1).unsqueeze(0).to(mps_device)

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image_tensor, text, return_tensors="pt")
    inputs = {key: tensor.to(mps_device) for key, tensor in inputs.items()}

    out = model.generate(**inputs)

    # unconditional image captioning
    inputs = processor(raw_image_tensor, return_tensors="pt")
    inputs = {key: tensor.to(mps_device) for key, tensor in inputs.items()}

    out = model.generate(**inputs)
    return (processor.decode(out[0], skip_special_tokens=True))

# Assuming 'decoded_image' contains the PIL image object
# print(BLIP_Generate_Caption(dataset["testmini"][12]['decoded_image']))

def BLIP_Generate_Caption_from_URL(image_url):
    # Fetch image from URL
    response = requests.get(image_url)
    image_data = Image.open(BytesIO(response.content))

    # Convert image data to tensor
    raw_image = image_data.convert('RGB')
    raw_image_tensor = torch.tensor(np.array(raw_image)).permute(2, 0, 1).unsqueeze(0).to(mps_device)

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image_tensor, text, return_tensors="pt")
    inputs = {key: tensor.to(mps_device) for key, tensor in inputs.items()}
    out = model.generate(**inputs)

    # unconditional image captioning
    inputs = processor(raw_image_tensor, return_tensors="pt")
    inputs = {key: tensor.to(mps_device) for key, tensor in inputs.items()}
    out = model.generate(**inputs)

    return (processor.decode(out[0], skip_special_tokens=True))
         

#def prediction_mode():
    # print ("Enter 1 if only the question is passed                  : ")
    # print ("Enter 2 if only the image URL is passed                 : ")
    # print ("Enter 3 if both Question and Image URL is to be passed  : ")
    # print ("Enter 4 to exit program                                 : ")
    # choice = int(input("Enter choice :"))
    # if (choice not in [1,2,3]):
    #     print ("Enter a valid choice again: ")
    #     choice = prediction_mode()
    # else :
    #     return choice

#choice = prediction_mode()

def df_splitter_and_vectorizer(df):
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



def test_text_processor_for_onlyImage(text, answer_type, X_train_tfidf, y_train, result):
    #print (text)
    # Instantiate SVM classifier
    SVM = svm.SVC(C=10, kernel='linear', degree=2, gamma='auto')
    
    # Fit the SVM classifier on the training dataset
    SVM.fit(X_train_tfidf, y_train[answer_type])

    # Preprocess the input text
    preprocessed_text = data_cleaner.preprocessing_text(text)
    #print("Preprocessed text:", preprocessed_text)

    # Transform preprocessed text using tfidf_vectorizer
    preprocessed_text = tfidf_vectorizer.transform([preprocessed_text])

    # Predict the label
    prediction = SVM.predict(preprocessed_text)

    # Print the prediction
    result.append(prediction[0])
    #print("Prediction for "+answer_type+" :", prediction)
    return result


            
def predictor_for_URL( URL, X_train_tfidf, y_train):
    print (URL)
    response = requests.get(URL)
    img = Image.open(BytesIO(response.content))
    img.show()
    for answer_type in ['answer_type','metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language']:
        test_text_processor_for_onlyImage(BLIP_Generate_Caption_from_URL(URL), answer_type, X_train_tfidf, y_train)
        print ('\n')

def predictor_for_Question( Question, X_train_tfidf, y_train):

    for answer_type in ['answer_type','metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language']:
        test_text_processor_for_onlyImage(Question, answer_type, X_train_tfidf, y_train)
        print ('\n')



def predictor_for_Question_and_URL( Question, URL, X_train_tfidf, y_train):
    #print (URL)
    response = requests.get(URL)
    img = Image.open(BytesIO(response.content))
    img.show()
    result = []
    for answer_type in ['answer_type','metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language']:
        result = test_text_processor_for_onlyImage(BLIP_Generate_Caption_from_URL(URL)+" "+Question, answer_type, X_train_tfidf, y_train, result)
        #print ('\n')
    return result



def get_df_paths():



        #This Module will get the path of the 3 DATA FRAMES. 
        #It will return csv_path which is a list whose each element is path to the dataframe
        

     
    # Get the current directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    

    # Navigate one folder back to the parent directory
    #parent_directory = os.path.dirname(current_directory)
    

    # Navigate to the "data" folder inside the parent directory
    data_directory = os.path.join(parent_directory, "data")
    

    # Navigate to the "processed" subfolder inside the "data" folder
    processed_directory = os.path.join(data_directory, "processed")

    
    # Define the paths to the data files
    csv_file_paths = [
        os.path.join(processed_directory, "df_for_only_question.csv"),
        os.path.join(processed_directory, "df_for_image_captions.csv"),
        os.path.join(processed_directory, "df_for_image_captions+Questions.csv"),
    ]

    return csv_file_paths


def class_influence_checker(Question, URL):

        df = pd.read_csv(csv_file_paths[2])
        X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
        return (predictor_for_Question_and_URL( Question, URL, X_train_tfidf, y_train))




def class_predictor(Question, URL, choice):


    #choice = 3

    if (choice == 1):
        df = pd.read_csv(csv_file_paths[0])
        X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
        Question = input("Enter Quesion: ")       
        print ("\n")
        predictor_for_Question( Question, X_train_tfidf, y_train)

    elif (choice == 2):
        df = pd.read_csv(csv_file_paths[1])
        X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
        print('Enter Image URL (press Enter to skip): ')
        URL = input().strip('"')
        print ("\n")
        predictor_for_URL( URL, X_train_tfidf, y_train)   
        
    elif (choice == 3):
        df = pd.read_csv(csv_file_paths[2])
        X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
        #print('Enter Image URL (press Enter to skip): ')
        #Question = input("Enter Quesion")
        #URL = input('Enter Image URL (press Enter to skip): ').strip('"')
        #print ("\n")
        return (predictor_for_Question_and_URL( Question, URL, X_train_tfidf, y_train))


    else:
        exit()    




  

csv_file_paths = get_df_paths()

# class_predictor()

# QUESTION : WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE 
# QUESTION : WHAT IS THE AREA OF THE FIGURE SHOWN BELOW. 


# HORSE PULLING A CAR : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIkJDpcdIMFHRjsgvc2JD0RspkUStSXuoIBQ&usqp=CAU
# CAT GIVING HIGH FIVE TO STAUTE OF LIBERTY : https://randomwordgenerator.com/img/picture-generator/52e1d5424b56aa14f1dc8460962e33791c3ad6e04e50744074267bd69149c7_640.jpg
# DIES IMAGE : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzqqxGuOfwNRD61afNt4iX0eBmvcZfCWx5Tg&usqp=CAU
# FOOD CHAIN : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO-8eWbdZ5aEbz1PJ1ryIFYuuG18u8tCc0YXxg33av7SatC5Er8TK9PplEa6IdXsTbQFQ&usqp=CAU


# MAP QUESTION : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSY3NVSNmEeGeEcwByTywefdCi2fncYFkgPcA&usqp=CAU
# PUZZLE PROBLEM : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpcIq0dBVcbISQ9c9lIcqFtd7OnUy9U5reFg&usqp=CAU  # Slight miss classification because of word cones.




