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
        words_name_type = np.zeros((dims[0], 2), dtype=object)
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
        words = bare_text.lower().split(' ')
    return words


def vectorize_text(list_of_text_files):
    """
        Vectorize the words in the text files
    :param
        list_of_text_files (list of strings): List of file names for the text files contained within the directory
    :return:
        vector_rep_array (numpy array of float 32): Array containing the expansion of words in the 300-dimensional basis
    """
    # Initialize and empty list for the vector representation of
    vector_rep_files = []
    for text_file in list_of_text_files:
        words = load_text_file(os.path.join(text_files_folder, text_file))
        # Put the words into a vector
        aux_list = []
        # This for loop runs once for each text file
        for word in words:
            idxs_array = np.where(words_name_type[:, 0] == word)
            idxs = idxs_array[0]
            if len(idxs) != 0:
                # First occurrence is taken for now
                aux_list.append(words_rep_vec[idxs[0]])
        # print(text_file)
        vector_rep_files.append(np.array(aux_list))
        max_dim = max([text_array.shape[0] for text_array in vector_rep_files])
        dim_equal = [np.pad(vec, ((0, max_dim - vec.shape[0]), (0, 0)), mode='constant') for vec in vector_rep_files]
        vector_rep_array = np.array(dim_equal)
    return vector_rep_array


def vectorize_search(search_keywords):
    """
        Return the search keywords in numpy array form for multiplication
    :param
        search_keywords (list of strings): list of keywords to be searched in the documents
    :return:
        array_rep (numpy array of float32): array representation of the search keywords in 300-word basis
    """
    num_words = len(search_keywords)
    array_rep = np.empty((num_words, 300), dtype=np.float32)
    for i, word in enumerate(search_keywords):
        idxs_array = np.where(words_name_type[:, 0] == word)
        idxs = idxs_array[0]
        if len(idxs) != 0:
            # First occurrence is taken for now
            array_rep[i, :] = words_rep_vec[idxs[0]]
    return array_rep


# Hyper-parameters for the code to work properly
model_location = './0/model.txt'  # file containing the model parameters
text_files_folder = './example_texts'  # folder containing the text files
search_keywords = ['banana']

words_rep_vec, words_name_type = load_model(model_location)
list_of_text_files = np.array(os.listdir(text_files_folder))
text_arrays_from_files = vectorize_text(list_of_text_files)
search_array = vectorize_search(search_keywords)
# Inner product to provide a metric for the presence of the search term
# Shape (#search keywords, # docs, max # of words in document)
inner_prod = np.inner(search_array, text_arrays_from_files)
# Get the max values for each document
max_vals = inner_prod.max(axis=2).squeeze()
sort_ind = np.flip(max_vals.argsort())  # In decreasing order
print("In order of decreasing prevalence of the keyword")
print(list_of_text_files[sort_ind])


