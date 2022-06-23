import numpy as np


def load_model(text_file_path):
    """
    :param
        text_file_path (str): The relative path to the text file where the model is loaded from
    :return:
        words_vec (numpy array of float32): Representation of words in the model
        words_name_type (numpy array of string): Word and type of word
    """
    with open(text_file_path, "r", encoding='utf8') as file:
        # Read the first line to get the dimensions of the vocabulary and word vector
        first_line = file.readline()
        dims = first_line.replace('\n', '').split(' ')
        dims = np.array(dims, dtype=int)
        # Allocate the memory in numpy
        words_vec = np.zeros((dims[0], dims[1]), dtype=np.float32)
        words_name_type = np.zeros((dims[0], 2), dtype=str)
        for idx, line in enumerate(file):
            line_list = line.replace('\n', '').split(' ')
            words_vec[idx, :] = line_list[1:]
            words_name_type[idx, :] = line_list[0].split('_')
            # input("Press to continue")
    return words_vec, words_name_type


words_vec, words_name_type = load_model('./0/model.txt')