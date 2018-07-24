import numpy as np
import os


def load_embeddings(directory, filename, embedding_dim):
    wordsList = []
    wordVectors = np.empty(embedding_dim)

    print("Loading word embeddings...")
    with open(os.path.join(directory, filename), 'r') as data:
        # f = data.read().encode("utf-8")
        i = 0
        # j = 0
        for line in data:
            values = line.split()
            word = values[0]
            # j += 1
            # print(str(j) + " " + line)
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                word = word.decode('UTF-8')
                wordsList.append(word)
                wordVectors = np.append(wordVectors, coefs)
            except ValueError:
                i += 1
    data.close()
    print('Loaded %s word vectors.' % len(wordsList))

    print(len(wordsList))
    print(wordVectors.shape)

    baseballIndex = wordsList.index('baseball')
    wordVectors[baseballIndex]

    return wordsList, wordVectors