from gensim.models import Word2Vec


def word_embeding():
    corpus = [["pisica", "stă", "pe", "covor"],
              ["câinele", "se", "joacă", "pe", "covor"]]

    model = Word2Vec(corpus, vector_size=3, window=2, min_count=1)
    print("Embedding pentru 'pisica':", model.wv["pisica"])
    print("Embedding pentru 'câinele", model.wv["câinele"])
    print("Embedding pentru 'covor", model.wv["covor"])


word_embeding()
