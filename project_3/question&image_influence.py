import testing_main
import word_influence
import os 
import sys
import importlib
import plotly.graph_objs as go

current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
src_data_directory = os.path.join(current_directory, "src", "data")
sys.path.append(src_data_directory)
import data_cleaner
importlib.reload(data_cleaner)
# File name is data_cleaner


#csv_file_paths = testing_main.get_df_paths()
##########################################################################################

def count_words(sentence):

    # Split the sentence into words using whitespace as delimiter
    words = sentence.split()
    # Count the number of words
    num_words = len(words)
    return num_words

##########################################################################################


def print_word_influence(word_influence_list, new_caption_len):
    for i, word_influence in enumerate(word_influence_list[:new_caption_len]):
        print(f"Word influence for class {i+1}:")
        for word, influence in word_influence.items():
            print(f"{word}: {influence}")
        print("-------------------------------")


# Example usage:
# question = "WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE " 
# image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIkJDpcdIMFHRjsgvc2JD0RspkUStSXuoIBQ&usqp=CAU"
# word_influence_list = word_influence.question_image_influence(question,image_url)
# print (word_influence_list)


# print_word_influence(word_influence_list, new_caption_len)



##########################################################################################

result_list = []

def question_image_influence_list (result_list, word_influence_list, question_len, caption_len):

    for i in range (6):

        # print(word_influence_list[i])
        sum_of_values_questions = sum(word_influence_list[i][key] for key in list(word_influence_list[i].keys())[:question_len])
        sum_of_values_captions = sum(word_influence_list[i][key] for key in list(word_influence_list[i].keys())[question_len:question_len+caption_len])
        
        temp_list = [sum_of_values_questions, sum_of_values_captions]

        result_list.append(temp_list)
        
    return result_list



##########################################################################################


def plot_question_image_influence(data):


    # Extracting x and y values for each group of bars
    x_values = ['Answer_Type', 'Metadata_Category', 'Metadata_Task', 'Metadata_Context', 'Metadata_Grade', 'Metadata_Language']
    y_values1 = [item[0] for item in data]  # Y values for the first bar in each group
    y_values2 = [item[1] for item in data]  # Y values for the second bar in each group

    # Creating traces for the two sets of bars
    trace1 = go.Bar(
        x=x_values,
        y=y_values1,
        name='Question Prompt'
    )

    trace2 = go.Bar(
        x=x_values,
        y=y_values2,
        name='Image Prompt'
    )

    # Combining traces into a data list
    data = [trace1, trace2]

    # Creating layout
    layout = go.Layout(
        title='Question and Image Influence',
        xaxis=dict(title='Groups'),
        yaxis=dict(title='Values'),
        barmode='group'
    )

    # Creating figure
    fig = go.Figure(data=data, layout=layout)

    # Displaying the plot
    fig.show()

##########################################################################################


# print (word_influence_list[1])
# print (question_image_influence (result_list, word_influence_list, question_len, caption_len))


def main():

    image_url = "https://randomwordgenerator.com/img/picture-generator/52e1d5424b56aa14f1dc8460962e33791c3ad6e04e50744074267bd69149c7_640.jpg"
    question =  "WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE "

    question_len = count_words(question)

    captions = testing_main.BLIP_Generate_Caption_from_URL(image_url)

    caption_len = count_words(captions)

    word_influence_list = word_influence.question_image_influence(question,image_url)

    # print_word_influence(word_influence_list, caption_len)
    # List of lists

    word_influence.class_predictor(" WHAT IS THE SURFACE INTEGRAL OF THE FOLLOWING FIGURE  " )

    data = question_image_influence_list (result_list, word_influence_list, question_len, caption_len)

    plot_question_image_influence(data)


##########################################################################################


main()