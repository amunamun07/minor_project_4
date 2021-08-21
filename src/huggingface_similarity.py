
from sentence_transformers import SentenceTransformer, util
import string

def huggingface_similar(sentences: list):
  """
  function: to find the similarity score between any two sentences
  parameters:
  sentence: a list of 2 user input sentences

  """
  #removing the grammar syntax from the sentences
  for i in range(0, len(sentences)):
    sentences[i] = sentences[i].translate(str.maketrans('', '', string.punctuation))

  #creating the model object using Sentencetransformer
  model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
  
  # encode list of sentences to get their embeddings
  embeddings = model.encode(sentences, convert_to_tensor=True)
  # compute similarity scores of two embeddings
  cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
  return cosine_scores.item()
  