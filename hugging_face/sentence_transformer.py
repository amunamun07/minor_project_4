from sentence_transformers import SentenceTransformer, util
import string


class ModelingSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        model = SentenceTransformer(self.model_name)
        return model

    def get_similarities(self, model, sentences):
        embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_scores.item()

