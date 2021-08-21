
from src.huggingface_similarity import *

sentence_list = [sen for sen in input("Input 2 sentences to find the similarity:\n").split('.') if len(sen) !=0]
assert len(sentence_list) == 2, "the function compares the similarity score of only 2 sentences at a time."
print("Sentence 1: {}".format(sentence_list[0]))
print("Sentence 2: {}".format(sentence_list[1]))

similarity_score = huggingface_similar(sentence_list)
print("Similarity score: {}".format(similarity_score))