from sentence_transformers import SentenceTransformer, util


class ModelingSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        """Loads the model in Sentence Transformer"""
        model = SentenceTransformer(self.model_name)
        return model

    @staticmethod
    def get_similarities(model, sentences):
        """Calculates the similarity probability and label for the sentences

        Args:
            model: a model object
            sentences: a list of sentences

        Returns:
            cosine_score: similarity score for given list of sentences
        """
        embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_scores.item()
