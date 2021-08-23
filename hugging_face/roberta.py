from src.predictions import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
import torch


class ModelingRoberta:
    def __init__(self, roberta_model_name):
        self.model_name = roberta_model_name

    def load_model(self):
        """Loads the pretrained model object and pretrained tokenizer object from hugging face"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return tokenizer, model

    @staticmethod
    def get_output(model, tokenizer, sentences):
        """Creates a object of RoBerta model and tokenizer after encoding the input sentences.

        Args:
            model: an model object
            tokenizer : a tokenizer object
            sentences: a list of sentences

        Returns:
            output: a model object after encoding input sentences
        """
        tokenized_input_seq_pair = tokenizer.encode_plus(
            sentences[0],
            sentences[1],
            max_length=256,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = (
            torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
        )
        token_type_ids = (
            torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
        )
        attention_mask = (
            torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
        )
        outputs = model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return outputs

    def get_prediction(self, model, tokenizer, sentences, label_mapping):
        """Calculates the similarity probability and label for the sentences

        Args:
            model: an model object
            tokenizer : a tokenizer object
            sentences: a list of sentences
            label_mapping: a list of sentences

        Returns:
            probability_percentage: model confidence in percentage
            predicted_label: label of the prediction
        """
        outputs = self.get_output(model, tokenizer, sentences)

        predicted_probability = torch.softmax(outputs[0], dim=1)[0]
        predicted_index = torch.argmax(predicted_probability)
        predicted_probability = predicted_probability.tolist()

        probability_percentage = predicted_probability[int(predicted_index)] * 100
        predicted_label = label_mapping[int(predicted_index)]

        return probability_percentage, predicted_label

    @staticmethod
    def evaluate(tokenizer, model, label_mapping, df_test):
        """Calculates the similarity probability and label for the sentences

        Args:
            model: an model object
            tokenizer : a tokenizer object
            label_mapping: a list of sentences
            df_test: a dataframe of test data

        Returns:
            total: total number of test data
            correct: total number of correct predictions
        """
        correct = 0
        total = 0
        for index, row in df_test.iterrows():
            total += 1
            predicted_probability, pred_index = get_prediction(
                tokenizer, model, row["sentence1"], row["sentence2"]
            )
            if row["similarity"] == label_mapping[int(pred_index)]:
                correct += 1

        return total, correct
