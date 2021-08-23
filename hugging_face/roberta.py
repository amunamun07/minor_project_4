from src.predictions import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
import torch


class ModelingRoberta:
    def __init__(self, roberta_model_name):
        self.model_name = roberta_model_name

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return tokenizer, model

    @staticmethod
    def get_output(model, tokenizer, sentences):
        tokenized_input_seq_pair = tokenizer.encode_plus(sentences[0], sentences[1],
                                                         max_length=256,
                                                         return_token_type_ids=True,
                                                         truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return outputs

    def get_prediction(self, model, tokenizer, sentences, label_mapping):
        tokenized_input_seq_pair = tokenizer.encode_plus(sentences[0], sentences[1],
                                                         max_length=256,
                                                         return_token_type_ids=True,
                                                         truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        predicted_probability = torch.softmax(outputs[0], dim=1)[0]
        predicted_index = torch.argmax(predicted_probability)
        predicted_probability = predicted_probability.tolist()

        probability_percentage = (predicted_probability[int(predicted_index)] * 100)
        predicted_label = label_mapping[int(predicted_index)]

        return probability_percentage, predicted_label

    @staticmethod
    def get_similarity(tokenizer, sentences):
        tokenized_input_1 = tokenizer.encode(sentences[0])
        input_id_1 = torch.Tensor(tokenized_input_1['input_ids']).long().unsqueeze(0)

        tokenized_input_2 = tokenizer.encode(sentences[1])
        input_id_2 = torch.Tensor(tokenized_input_2['input_ids']).long().unsqueeze(0)
        print(cosine_similarity(input_id_1, input_id_2))

    @staticmethod
    def evaluate(tokenizer, model, label_mapping, df_test):
        correct = 0
        total = 0
        for index, row in df_test.iterrows():
            total += 1
            predicted_probability, pred_index = get_prediction(tokenizer, model, row["sentence1"], row["sentence2"])
            if row["similarity"] == label_mapping[int(pred_index)]:
                correct += 1

        print('Out of {total} pair of sentences, {correct} were found to be correctly predicted. '
              'Calculated accuracy of this model is {correct}%'.format(total=total, correct=correct))
