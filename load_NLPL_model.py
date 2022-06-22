import numpy as np
import pandas as pd
import tqdm

with open("./word_vec_rep/model.txt", "r", encoding='utf8') as file:
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

# model = gensim.models.fasttext.load_facebook_model("./word_vec_rep/model.bin")
# pass