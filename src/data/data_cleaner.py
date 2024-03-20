from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



def preprocessing_text(text):

    """
        Handles basic text processing
    """
    # Make Lower case all character in text
    text = text.lower()                                         # Removes non alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)                        # Tokenizing words
    tokens = word_tokenize(text)                                # Obtaining stop words like ‘a’, ‘the’ etc.
    stop_words = set(stopwords.words('english'))                # Removing stop words lie (a, the, is)
    tokens = [word for word in tokens if word not in stop_words]# Obtaining base of words like ‘happy’ and not ‘happiest’
    lemmatizer = WordNetLemmatizer()                            # lemmatizing each word.
    tokens = [lemmatizer.lemmatize(word) for word in tokens]    # Joining and returning
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def detect_language(text):
    # Function to detect the language of the given text
    try:
        # Attempt to detect the language
        lang = detect(text)
        return lang
    except:
        # Return None if an exception occurs (e.g., unsupported language)
        return None


def remove_non_english(df):
    """
        Removes non english words
    """

    # Apply language detection to each text entry in the DataFrame
    df['language'] = df['preprocessed_text'].apply(detect_language)

    # Filter out non-English entries
    df = df[df['language'] == 'en'].reset_index(drop=True)

    # Drop the language column as it's no longer needed
    df.drop(columns=['language'], inplace=True)

    return df


# Split the data into train and test sets

def train_test_splitter(df):

    """
        # Splits data in train and test
    
    """


    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], 
                                                        df[['metadata_language', 'metadata_skills', 'metadata_task', 
                                                            'metadata_category', 'metadata_context', 'metadata_grade',
                                                            'metadata_split', 'metadata_source', 'metadata_img_height', 
                                                            'metadata_img_width']], 
                                                        test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test
    


def TF_IDF_vectorising_function(X_train, X_test):
    """
        Performs TF-IDF vectorisation on X_train and X_test
    
    """
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the preprocessed text data and transform it
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) < 2:
                continue  # Skip lines with insufficient elements
            word = values[0]
            try:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
            except ValueError:
                continue  # Skip lines with invalid embeddings
    return embeddings

    # Path to the GloVe file
    # glove_file_path = '/Users/psinha/Downloads/glove.840B.300d.txt'

    # Load GloVe embeddings
    # glove_embeddings = load_glove_embeddings(glove_file_path)

def get_glove_embeddings(sentence,glove_embeddings):
    embeddings = []
    for word in sentence.split():
        if word in glove_embeddings:
            embeddings.append(glove_embeddings[word])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(300)  # Assuming GloVe vectors are of size 300

# Apply GloVe embeddings to preprocessed text
#X_train_glove = np.array([get_glove_embeddings(text) for text in X_train])
#X_test_glove = np.array([get_glove_embeddings(text) for text in X_test])