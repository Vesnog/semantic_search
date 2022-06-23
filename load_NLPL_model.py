import numpy as np
import string
import os


def load_model(model_file_path):
    """
        Loads the word model from a text file
    :param
        model_file_path (str): The relative path to the text file where the model is loaded from
    :return:
        words_rep_vec (numpy array of float32): Representation of words in the model
        words_name_type (numpy array of string): Word and type of word
    """
    with open(model_file_path, "r", encoding='utf8') as file:
        # Read the first line to get the dimensions of the vocabulary and word vector
        first_line = file.readline()
        dims = first_line.replace('\n', '').split(' ')
        dims = np.array(dims, dtype=int)
        # Allocate the memory in numpy
        words_rep_vec = np.zeros((dims[0], dims[1]), dtype=np.float32)
        # Use the data type object as this is suitable for variable length strings
        words_name_type = np.zeros((dims[0], 2), dtype=np.object)
        for idx, line in enumerate(file):
            line_list = line.replace('\n', '').split(' ')
            words_rep_vec[idx, :] = line_list[1:]
            words_name_type[idx, :] = line_list[0].split('_')
            # input("Press to continue")
    return words_rep_vec, words_name_type


def load_text_file(text_file_path):
    """
        Loads a text file, parses and cleans it returning the words as elements of a numpy array
    :param
        text_file_path (str): The relative path to the text file containing text to be searched
    :return:
        words_vec (numpy array of object): Words contained in the text file
    """
    with open(text_file_path, "r", encoding='utf8') as file:
        all_text = file.read()
        # Get rid of the punctuation first
        no_punc = all_text.translate(str.maketrans('', '', string.punctuation))
        # Get rid of the new lines and trim both ends by removing leading and trailing white space
        bare_text = no_punc.replace('\n', ' ').strip()
        words = bare_text.split(' ')
    return words


# Hyper-parameters for the code to work properly
model_location = './0/model.txt'  # file containing the model parameters
text_files_folder = './example_texts'  # folder containing the text files

words_rep_vec, words_name_type = load_model(model_location)
list_of_text_files = os.listdir(text_files_folder)
for text_file in list_of_text_files:
    words = load_text_file(os.path.join(text_files_folder, text_file))

pass


