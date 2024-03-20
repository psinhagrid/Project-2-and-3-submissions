import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import datasets
import os
import io
import joblib
import ast
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from datasets import load_dataset
import datasets
import pandas as pd
import re


from sklearn.decomposition import TruncatedSVD  # LSA is essentially truncated SVD
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from langdetect import detect

from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn import svm, naive_bayes, metrics
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

# Disable the specific UserWarning
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# Performing TFIDF vectorisation
tfidf_vectorizer = TfidfVectorizer()


##########################################################################################

def df_splitter_and_vectorizer(df):
            
            """

                Function to split data into train test, and also vectorize the same
                Data Frame is passed and X_train, X_test, y_train, y_test is returned

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


##########################################################################################


def get_df_paths():

    """

        This Module will get the path of the 3 DATA FRAMES. 
        It will return csv_path which is a list whose each element is path to the dataframe
        
    """
     
    # Get the current directory
    current_directory = os.getcwd()

    # Navigate one folder back to the parent directory
    parent_directory = os.path.dirname(current_directory)
    

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

csv_file_paths = get_df_paths()


##########################################################################################

def data_frame_loader():
        """

            This function will load the dataframe according to the user demand. 
            The output will be the Data Frame. 

        """

        # We will take user input to decide what type of input is required to assess.
        print("Enter 1 if only question given              :")
        print("Enter 2 if only image given                 :")
        print("Enter 3 if both question and image given    :")
        choice = int(input("ENTER YOUR INPUT TYPE          :"))



        # We will read the corresponding CSV file to user input. 
        if choice == 1:
            df = pd.read_csv(csv_file_paths[0])
        elif choice == 2:
            df = pd.read_csv(csv_file_paths[1])
        elif choice == 3:
            df = pd.read_csv(csv_file_paths[2])
        else:
            print("INVALID CHOICE ")

        return df

df = data_frame_loader()

# Now we can split and vectorise our data. 
X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)

print ("Data Frame Obtained")



##########################################################################################

# Now, we will try to tune our Random forest model

def best_max_depth_for_forest(max_depth_list):
    """
        This function will find out the best value of max depth. 
        After running this, we will see that the best value of max_depth is as high as possible. We choose 20.
    """
    # Create a RandomForestClassifier object
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'max_depth': max_depth_list  # You can adjust the range of max_depth values to search
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

    # Perform grid search on your data
    grid_search.fit(X_train_tfidf, y_train['metadata_task'])  # Assuming X_train_vectorised and y_train are defined

    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_score)

    # Train a RandomForestClassifier with the best parameters
    best_rf_classifier = RandomForestClassifier(max_depth=best_params['max_depth'], random_state=42)
    best_rf_classifier.fit(X_train_tfidf, y_train['metadata_task'])

    # Evaluate the classifier on the test set
    y_pred = best_rf_classifier.predict(X_test_tfidf)  # Assuming X_test_vectorised is defined
    test_accuracy = accuracy_score(y_test['metadata_task'], y_pred)  # Assuming y_test is defined
    print("Test Accuracy with Best Parameters:", test_accuracy)



    
def best_No_of_estimators():

    """
        This functions will help us choose how many trees should we keep in out forest.
        It will plot a graph, to help us choose.
        We see after 20 trees, there is no significant increase in accuracy of our validation set. 
    """
    rf_classifier = RandomForestClassifier(random_state=42, max_depth=50)

    # Define a range of values for n_estimators
    param_range = [10, 20, 50, 100, 200]  # You can adjust this range as needed

    # Filter the target variable for the specific class
    y_train_class = y_train['metadata_task']

    # Initialize the Random Forest classifier
    #rf_classifier = RandomForestClassifier(random_state=42)

    # Calculate training and validation scores using validation_curve
    train_scores, valid_scores = validation_curve(
        rf_classifier, X_train_tfidf, y_train_class, 
        param_name="n_estimators", param_range=param_range, 
        scoring="accuracy", cv=5  # You can adjust the number of cross-validation folds (cv) as needed
    )

    # Calculate mean and standard deviation of training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(param_range, valid_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15, color='green')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Validation Curve for metadata_task')
    plt.legend(loc='lower right')
    plt.xticks(param_range)
    plt.grid()
    plt.show()


##########################################################################################

def classifier_accuracy(rf_classifier, category):
    # Train the classifier
    rf_classifier.fit(X_train_tfidf, y_train[category])

    # Predict the labels for the test set
    y_pred = rf_classifier.predict(X_test_tfidf)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test[category], y_pred)
    print("Accuracy of " + category + ":", accuracy)




##########################################################################################
from sklearn.model_selection import GridSearchCV
from sklearn import svm, naive_bayes, metrics
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def best_SVM_Hyperparameters(model_name, accuracy_metric):

    # Define metadata unique classes
    metadata_task_unique_class = ['figure question answering','visual question answering','math word problem','geometry problem solving','textbook question answering']
    metadata_category_unique_class = ['general-vqa','math-targeted-vqa']
    answer_type_unique_class = ['float','text','list','integer']
    metadata_context_unique_class = ['scatter plot','synthetic scene','table','geometry diagram','bar chart','abstract scene','function plot','line plot','natural image','puzzle test','scientific figure','pie chart','map chart','medical image','document image','heatmap chart']
    metadata_grade_unique_class = ['daily life','elementary school','high school','college']
    metadata_language_unique_class = ['chinese','english','persian']

    metadata_features_list     = ['answer_type','metadata_category','metadata_task','metadata_context','metadata_grade','metadata_language']
    metadata_unique_class_list = [answer_type_unique_class,metadata_category_unique_class,metadata_task_unique_class,metadata_context_unique_class,metadata_grade_unique_class,metadata_language_unique_class]

    # Initialize the best model and score variables
    best_model = None
    best_score = -1

    # We will now get ready to apply grid search for best parameters.
    for i in range(len(metadata_features_list)):
        param_grid = {}
        if model_name == "SVM":
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf'], 'degree': [1, 2, 3, 4]}
            model = svm.SVC()
        elif model_name == "NB":
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
            model = naive_bayes.MultinomialNB()

        print("\nGrid search started for", metadata_features_list[i])
        grid_search = GridSearchCV(model, param_grid, scoring=accuracy_metric, cv=5)
        grid_search.fit(X_train_tfidf, y_train[metadata_features_list[i]])
        best_params = grid_search.best_params_
        best_score_feature = grid_search.best_score_
        print("Best Parameters for", metadata_features_list[i], ":", best_params)
        print("Best", accuracy_metric, "for", metadata_features_list[i], ":", best_score_feature)

        # Update best score and model if the current feature's best score is better
        if best_score_feature > best_score:
            best_score = best_score_feature
            best_model = grid_search.best_estimator_

    # Fit the best model on the entire training data (you can choose any metadata feature for fitting)
    best_model.fit(X_train_tfidf, y_train["metadata_task"])

    # Predict on the entire test set (you can choose any metadata feature for prediction)
    predictions_all = best_model.predict(X_test_tfidf)

    # Calculate overall accuracy (you can choose any metadata feature for calculating accuracy)
    overall_accuracy = accuracy_score(predictions_all, y_test["metadata_task"])
    print("Overall Accuracy:", overall_accuracy)

    # Compare with best score obtained from grid search
    print("Best Score from Grid Search:", best_score)

  
# Example usage
# X_train_tfidf, y_train, X_test_tfidf, y_test should be defined before calling the function
# best_SVM_Hyperparameters("SVM", "accuracy", X_train_tfidf, y_train, X_test_tfidf, y_test)

##########################################################################################         

def plot_accuracy_vs_degree(class_name):
    degrees = [ 1, 2, 3, 4, 5]  # Degrees of the polynomial kernel
    accuracies = []

    for degree in degrees:
        model = svm.SVC(kernel='poly', degree=degree)
        model.fit(X_train_tfidf, y_train[class_name])
        predictions = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test[class_name], predictions)
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, accuracies, marker='o')
    plt.title('Accuracy vs. Degree for SVM with Polynomial Kernel')
    plt.xlabel('Degree')
    plt.ylabel('Accuracy')
    plt.xticks(degrees)
    plt.grid(True)
    plt.show()

##########################################################################################         


def SVM_plotter():


    metadata_grade_unique_class = ['daily life', 'elementary school', 'high school', 'college']
    # metadata_task_unique_class = ['figure question answering', 'visual question answering', 'math word problem', 'geometry problem solving', 'textbook question answering']

    # Reduce dimensionality using LSA
    lsa = TruncatedSVD(n_components=2)
    X_train_lsa = lsa.fit_transform(X_train_tfidf)

    # Define a range of C values to test
    C_values = [0.01, 1, 100]

    # Plot decision boundaries for different C values
    plt.figure(figsize=(12, 8))

    for i, C in enumerate(C_values):
        # Train SVM classifier
        svm_classifier = svm.SVC(kernel='linear', C=C)
        svm_classifier.fit(X_train_lsa, y_train["metadata_grade"])
        
        # Plot decision boundary
        plt.subplot(2, 3, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Plot decision boundary
        xx, yy = np.meshgrid(np.linspace(-1, 1, 500),
                            np.linspace(-1, 1, 500))
        Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape[0], xx.shape[1], -1)

        # Plot decision boundaries for each pair of classes
        for idx1 in range(len(metadata_grade_unique_class)):
            for idx2 in range(idx1+1, len(metadata_grade_unique_class)):
                plt.contourf(xx, yy, Z[:,:,idx1] - Z[:,:,idx2], cmap=plt.cm.coolwarm, alpha=0.8)

        # Encode the string labels into numerical values
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train["metadata_grade"])

        # Plot data points
        plt.scatter(X_train_lsa[:, 0], X_train_lsa[:, 1], c=y_train_encoded, cmap='viridis', s=20, edgecolor='k')

    # Labeling the graphs
    for i, ax in enumerate(plt.gcf().get_axes(), 1):
        ax.set_title(f'C={C_values[i-1]}')
        ax.set_xlabel('LSA Component 1')
        ax.set_ylabel('LSA Component 2')

    plt.show()



##########################################################################################         
def model_trainer_svm(log_file="train_logs.txt"):
    """

    Trains SVM model and saves logs and models in respective files. 

    """
    metadata_features_list = ['answer_type', 'metadata_category', 'metadata_task', 'metadata_context', 'metadata_grade', 'metadata_language']
    svm_model_filenames = []

    # Create the directory for saving models if it doesn't exist
    model_save_dir = os.path.join(os.getcwd(), "MODEL_SAVED")
    os.makedirs(model_save_dir, exist_ok=True)

    # Create the directory for saving logs if it doesn't exist
    train_save_dir = os.path.join(os.getcwd(), "LOGS")
    os.makedirs(train_save_dir, exist_ok=True)

    # Open the log file in append mode using the full path
    log_file_path = os.path.join(train_save_dir, log_file)
    try:
        with open(log_file_path, "a") as f:
            for feature in metadata_features_list:
                f.write(f"Training SVM model for {feature}...\n")
                print(f"Training SVM model for {feature}...")
                
                # Create SVM model
                SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
                
                # Fit model to training data
                SVM.fit(X_train_tfidf, y_train[feature])
                
                # Calculate accuracy on training data
                train_accuracy = accuracy_score(y_train[feature], SVM.predict(X_train_tfidf))
                
                # Log accuracy
                f.write(f"Training accuracy for {feature}: {train_accuracy:.4f}\n")
                print(f"Training accuracy for {feature}: {train_accuracy:.4f}")
                
                #f.write(f"SVM model for {feature} trained.\n")
                #print(f"SVM model for {feature} trained.")
                
                # Generate a unique filename for saving the model
                filename = os.path.join(model_save_dir, f"SVM_model_Question_&_Images_{feature}.joblib")
                
                # Save the model
                joblib.dump(SVM, filename)
                svm_model_filenames.append(filename)
                

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

    return svm_model_filenames

# Example usage:
# log_file_path = "train_logs_Question_&_Images_only.txt"
# svm_model_files = model_trainer_svm(log_file=log_file_path)
# if svm_model_files:
#     print("SVM model filenames:", svm_model_files)
# else:
#     print("Training failed. Check logs for details.")


########################################################################################## 

# Model Evaluation


# Now we will define a function to evaluate model
def evaluation_of_model(model_name, accuracy_metric):

    """
    This function takes in model name and accuracy matrix and return the accuracy/confusion matrix
    
    """

    # First we make a list if unique elements in classes

    metadata_task_unique_class = ['figure question answering','visual question answering','math word problem','geometry problem solving','textbook question answering']
    metadata_category_unique_class = ['general-vqa','math-targeted-vqa']
    answer_type_unique_class = ['float','text','list','integer']
    metadata_context_unique_class = ['scatter plot','synthetic scene','table','geometry diagram','bar chart','abstract scene','function plot','line plot','natural image','puzzle test','scientific figure','pie chart','map chart','medical image','document image','heatmap chart']
    metadata_grade_unique_class = ['daily life','elementary school','high school','college']
    metadata_language_unique_class = ['chinese','english','persian']

    metadata_features_list     = ['anwer_type','metadata_category','metadata_task','metadata_context','metadata_grade','metadata_language']
    metadata_unique_class_list = [answer_type_unique_class,metadata_category_unique_class,metadata_task_unique_class,metadata_context_unique_class,metadata_grade_unique_class,metadata_language_unique_class]


    metadata_features_list     = ['answer_type','metadata_category','metadata_task','metadata_context','metadata_grade','metadata_language']
    metadata_unique_class_list = [answer_type_unique_class,metadata_category_unique_class,metadata_task_unique_class,metadata_context_unique_class,metadata_grade_unique_class,metadata_language_unique_class]




   
    X_train_tfidf, X_test_tfidf, y_train, y_test = df_splitter_and_vectorizer(df)
 
    for i in range(len(metadata_features_list)):

        if (model_name == "SVM"): 
            # Fit the training dataset on the SVM classifier
            SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
            SVM.fit(X_train_tfidf, y_train[metadata_features_list[i]])

            # Predict the labels on the validation dataset
            predictions = SVM.predict(X_test_tfidf)

        elif (model_name == "NB"):
            Naive = naive_bayes.MultinomialNB()
            Naive.fit(X_train_tfidf,y_train[metadata_features_list[i]])
            
            # predict the labels on validation dataset
            predictions = Naive.predict(X_test_tfidf)

        

        if (accuracy_metric == "confusion_metric"):

            

            confusion_matrix = metrics.confusion_matrix(y_test[metadata_features_list[i]], predictions)

            cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=metadata_unique_class_list[i])

            fig, ax = plt.subplots()
            cm_display.plot(ax=ax, xticks_rotation='vertical')  # Rotating x-axis labels vertically
            ax.set_title("Confusion Matrix For "+metadata_features_list[i])  # Setting the title

            # Print classification report
            report = classification_report(y_test[metadata_features_list[i]], predictions, target_names=metadata_unique_class_list[i])
            print("\nClassification Report For "+metadata_features_list[i]+"\n", report)

            plt.show()

        elif (accuracy_metric == "accuracy"):

            print("\n Accuracy of -> "+metadata_features_list[i],accuracy_score(predictions, y_test[metadata_features_list[i]])*100)




