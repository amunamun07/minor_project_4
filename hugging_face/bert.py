from src.predictions import *
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class ModelingBert:
    def __init__(self, model_name):
        self.model_name = model_name


    def load_model(self):
        """
        :return: tokenizer: tokenizer has 3 values:
            - input_ids. The list/tensor of token ids for each sentence. The id 101 represents the class token [CLS],
                            102 represents the separator token [SEP], and 0 represents the padding token [PAD].
            - attention_mask. This list/tensor represents which ids to use when generating the tokens (e.g. ignores the
                                [PAD] tokens).
            - token_type_ids. This list/tensor represents which tokens correspond to the first and second sentence
                                (used for next sentence prediction).
        :return: model:
        """
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name)
        return tokenizer, model

    def get_similarities(self, tokenizer, model, sentences):
        """
        # pad the texts to the maximum length (so that all outputs have the same length)
        # return the tensors (not lists)

        :param tokenizer:
        :param model:
        :param sentences:
        :return:
        """
        # encodings = tokenizer(sentences, padding=True, return_tensors='pt')
        # with torch.no_grad():
        #     embeds = model(**encodings)
        # embeddings = model.encode()
        # for sentence in sentences:
        #     new_tokens = tokenizer.encode_plus()
        # return cosine_similarity(sentences)

        tokenized_input_seq_pair = tokenizer.encode_plus(sentences[0], sentences[1],
                                                              max_length=256,
                                                              return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=None)

        predicted_probability = torch.softmax(outputs[0], dim=1)[0]  # batch_size only one
        predicted_index = torch.argmax(predicted_probability)

        probability_percentage = (predicted_probability[int(predicted_index)] * 100).tolist()
        predicted_label = label_mapping[int(predicted_index)]
        return probability_percentage, predicted_label
