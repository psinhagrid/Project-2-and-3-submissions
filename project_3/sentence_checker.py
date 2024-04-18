import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import os
import sys
import importlib

import nltk
import ssl



current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
src_data_directory = os.path.join(current_directory, "src", "data")
sys.path.append(src_data_directory)
import data_cleaner
importlib.reload(data_cleaner)
# File name is data_cleaner



def check_words(sentence):
    """
        The function to see if the sentence is composed of valid words
    
    """
    words = nltk.word_tokenize(sentence)
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    for word in words:
        if word.lower() not in english_vocab:
            return False
    
    return True


def sentence_checking(original_sentence):

    """
    This function preprocesses the sentence and then returns if the words are valid or invalid. 
    Uses check words function. 
    
    """
    processed_sentence = data_cleaner.preprocessing_text(original_sentence) # Processing using our preprocessing text file. 

    if not processed_sentence.strip():      # If processed sentence empty, returns invalid prompt
        return "invalid sentence"

    if check_words(processed_sentence):     # Checks if each word makes sense. 
        return "valid sentence"
    else:
        return "invalid sentence"




def URL_check(link):

    """
    Check if the link starts with "https".
    
    Parameters:
        link (str): The link to be checked.
    
    Returns:
        bool: True if the link starts with "https", False otherwise.
    """

    if (link.startswith("https") == False):
        return "insecure link"
    
    if not link.endswith(("jpg", "jpeg", "png", "CAU")):
        return "invalid link format"
    
    else :
        return "valid link"

    








