import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


class TFIDF:
    def __init__(self):
        pass

    def create_dict(self, sentences: list):
        """Creates dictionary (a bag of words) where each words(keys) are unique words and values:frequency(words)

        Args:
            sentences: a list of sentences

        Returns:
            dic: a dictionary of unique words from sentences
        """
        dic = {}
        for sentence in sentences:
            sentence = sentence.translate(str.maketrans("", "", string.punctuation))
            words = word_tokenize(sentence.lower())
            filtered_sentence = [w for w in words if not w in stop_words]
            for word in filtered_sentence:
                if word in dic.keys():
                    dic[word] = dic[word] + 1
                else:
                    dic[word] = 1
        return dic

    def tfidf_cosine_similarity(self, sentences):
        """Compute the term frequency and inverse document frequency matrix of given sentences and calculate the cosine
            similarity score.

        Args:
            sentences: a list of sentences

        Returns:
            cosine similarity score between sentences
        """
        bow_dic = self.create_dict(sentences)

        # Compute sentence term frequency matrix as well idf matrix for each term
        sentence_tf_matrix = np.zeros((len(sentences), len(bow_dic)))
        sentence_idf_matrix = np.zeros((len(bow_dic), len(sentences)))
        sentence_term_df = pd.DataFrame(
            sentence_tf_matrix, columns=sorted(bow_dic.keys())
        )
        sentence_count = 0
        sentence_list = []
        for sentence in sentences:
            sentence = sentence.translate(str.maketrans("", "", string.punctuation))
            words = word_tokenize(sentence.lower())
            for word in words:
                if word in bow_dic.keys():
                    sentence_term_df[word][sentence_count] = (
                        sentence_term_df[word][sentence_count] + 1
                    )
            sentence_count = sentence_count + 1
            sentence_list.append("sentence {}".format(sentence_count))

        # Computed idf for each word in vocab
        idf_dict = {}
        for column in sentence_term_df.columns:
            idf_dict[column] = (
                np.log(
                    (len(sentences) + 1) / (1 + (sentence_term_df[column] != 0).sum())
                )
                + 1
            )

        # compute tf.idf matrix
        sentence_tfidf_matrix = np.zeros((len(sentences), len(bow_dic)))
        sentence_tfidf_df = pd.DataFrame(
            sentence_tfidf_matrix,
            index=sorted(sentence_list),
            columns=sorted(bow_dic.keys()),
        )

        sentence_count = 0
        for sentence in sentences:
            for key in idf_dict.keys():
                sentence_tfidf_df[key][sentence_count] = (
                    sentence_term_df[key][sentence_count] * idf_dict[key]
                )
            sentence_count = sentence_count + 1

        # lets create a cosine similarity as a dataframe
        cosine_sim = cosine_similarity(sentence_tfidf_df, sentence_tfidf_df)
        cosine_sim_df = pd.DataFrame(
            cosine_sim, index=sorted(sentence_list), columns=sorted(sentence_list)
        )
        return cosine_sim_df
