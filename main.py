import logging
import yaml
import warnings
import pandas as pd
from hugging_face import ModelingRoberta
from hugging_face import ModelingSentenceTransformer
from numerical_representation import TFIDF

warnings.filterwarnings("ignore")


def helper_variables():
    """Initializing config.yaml object as config_file_instance,
        test data object as df_train and list of sentences as sentences.

    Returns:
        config_file_instance: a dict consisting of variables from config.yaml
        sentences: a list of sentences
        df_test: a dataframe of test data
    """

    with open("config.yaml", "r") as stream:
        config_file_instance = yaml.safe_load(stream)
    sentences = ["I love all the cats.", "I love all the dogs."]
    df_test = pd.read_csv(config_file_instance["testing_dataset_path"])

    return config_file_instance, sentences, df_test


def use_roberta_model(config_file_instance, sentences, df_test):
    """Using Roberta Model to make predictions

    Args:
        config_file_instance: a dict consisting of variables from config.yaml
        sentences: a list of sentences
        df_test: a dataframe of test data
    """
    roberta = ModelingRoberta(config_file_instance["roberta_model_name"])
    roberta_tokenizer, roberta_model = roberta.load_model()
    probability_percentage, predicted_label = roberta.get_prediction(
        roberta_model,
        roberta_tokenizer,
        sentences,
        config_file_instance["label_mapping"],
    )
    logging.info("==== Using RoBERTa Model ====")
    logging.info(
        "The sentences seem to have {probability_percentage:.2f}% probability to be {predicted_label}.".format(
            probability_percentage=probability_percentage,
            predicted_label=predicted_label,
        )
    )

    total_test_data, correct_predictions = roberta.evaluate(
        roberta_tokenizer, roberta_model, config_file_instance["label_mapping"], df_test
    )


def use_sentence_transformer_model(config_file_instance, sentences):
    """Using Sentence Transformer to calculate similarity

    Args:
        config_file_instance: a dict consisting of variables from config.yaml
        sentences: a list of sentences
    """
    sentence_model = ModelingSentenceTransformer(
        config_file_instance["sentence_transformer_model_name"]
    )
    model = sentence_model.load_model()
    sentence_transformer_similarity_score = sentence_model.get_similarities(
        model, sentences
    )
    logging.info("==== Using Sentence Transformer ====")
    logging.info("Similarities:{}".format(sentence_transformer_similarity_score))


if __name__ == "__main__":
    yaml_instance, sentence_corpus, testing_csv = helper_variables()

    use_roberta_model(yaml_instance, sentence_corpus, testing_csv)

    use_sentence_transformer_model(yaml_instance, sentence_corpus)

    TFIDF.tfidf_cosine_similarity(sentence_corpus)
