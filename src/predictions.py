import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import torch



def get_prediction(tokenizer, model, sentence_one, sentence_two, max_length=256):

    tokenized_input_seq_pair = tokenizer.encode_plus(sentence_one, sentence_two,
                                                    max_length=max_length,
                                                    return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0]  # batch_size only one
    predicted_index = torch.argmax(predicted_probability)
    predicted_probability = predicted_probability.tolist()

    return predicted_probability, predicted_index



def evaluate(tokenizer, model, label_mapping, df_test):
    correct = 0
    total = 0
    for index, row in df_test.iterrows():
        total += 1
        predicted_probability, pred_index = get_prediction(tokenizer, model, row["sentence1"], row["sentence2"])
        if row["similarity"] == label_mapping[int(pred_index)]:
            correct += 1

    print('Out of {total} pair of sentences, {correct} were found to be correctly predicted. Calculated accuracy of this model is {correct}%'.format(total=total, correct=correct))
    