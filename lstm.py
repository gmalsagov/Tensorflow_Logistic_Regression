import tensorflow as tf
from load_data import load_embeddings
import numpy as np

# directory = 'glove/glove.twitter.27B/'
directory = 'glove/glove.6B/'
# filename = 'glove.twitter.27B.200d.txt'
filename = 'glove.6B.100d.txt'
embedding_dim = 100

wordsList, wordVectors = load_embeddings(directory, filename, embedding_dim)

maxSeqLength = 10 #Maximum length of sentence
numDimensions = 100 #Dimensions for each word vector

firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence) #Shows the row index for each word


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)