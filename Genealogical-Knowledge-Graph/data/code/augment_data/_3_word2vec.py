from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import BrownCorpus,Text8Corpus
import nltk.tokenize
import gensim.downloader as api
from gensim.test.utils import datapath

# reference : https://segmentfault.com/a/1190000008173404?from=timeline
# path = get_tmpfile("word2vec.model")
# sentences = Text8Corpus('/home2/hk/workshop/Data/text8/text8')
# model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")

# directly use trained vec
# import gensim.downloader as api
# word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
# result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))

path = get_tmpfile("word2vec.model")


def get_word2vec_model():

    with open("../sentence_level_corpus_all_information_normalized.csv") as f:
        data = f.readlines()

    sentences = []
    for i in data[1:]:
        sen = i.split("|")[1]
        words = nltk.word_tokenize(sen)
        sentences.append(words)

    sentences_1 = [i for i in Text8Corpus('/home2/hk/workshop/Data/text8/text8')]
    sentences = sentences + sentences_1

    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=5, iter=10)
    model.wv.save(path)
    # model.save("word2vec.model")
    vector = model.wv['brother-in-law']  # numpy vector of a word
    print(vector)


def get_sim_word(key, word_list):
    word_vectors = KeyedVectors.load(path, mmap='r')
    # word_vectors = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-data

    similarity_list = []
    word = word_list[0]
    for i in word_list:
        similarity = word_vectors.similarity(key, i)
        similarity_list.append(similarity)
        if similarity >= max(similarity_list):
            word = i

    return word


if __name__ == "__main__":
    # get_word2vec_model()
    print(get_sim_word("grandchild", ["grandchildren", "dog", "weather"]))
