
# Minor_Project_4

The objective for this project is to derive similarity score using but hugging face(Sentence transformer / RoBERTa) models and numerical representation(tfidf).

### Code Structure
```
|
|   .gitignore
│   config.yaml
│   main.py
│   Pipfile
│   README.md
│
├───dataset
│       snli_testing.csv
│
├───hugging_face
│   │   roberta.py
│   │   sentence_transformer.py
│   │   __init__.py
│   │
│   └───__pycache__
│           bert.cpython-36.pyc
│           roberta.cpython-36.pyc
│           sentence_transformer.cpython-36.pyc
│           __init__.cpython-36.pyc
│
├───Notebook
│       roberta.ipynb
│       tf_idf_similarity_score.ipynb
│
├───numerical_representation
│       tfidf.py
│       __init__.py
│
└───src
        huggingface_similarity.py
        predictions.py
```

### Folders/files Description:
#### 1. Notebook
The Notebooks files were created to quickly view the outputs.
- Contains roberta.ipnyb which is a step by step process of finding probability of any two sentences to be similar(entailment, contradiction and neutral). It also contains description of Transformers and Bert.
- Contains tf_idf_similarity_score.ipnyb which outputs the similarity score between many sentences in a tabular form. 

#### 2. Hugging Face
- Contains roberta.py which is a py version of above mentioned notebook.
- Contains sentence_transformer which is used to find cosine similarity score between two sentences.

#### 3. Numerical Representation
- Contains tfidf.py which is a py version of the above mentioned notebook.

#### 4. Config.yaml
- has global variables,datapaths and dictionaries

#### 5. main.py
- has three functions to use predictions from roberta model, similarity from sentence_transformer and similarity from tfidf.

### How to run this repository:
Pre-requisits: Install pipenv(sudo apt-get pipenv)
#### Step 1: Setup 
- Clone the repository.
- pipenv shell

#### Step 2: Setup folders
- Create a dataset folder
- Add dataset from https://drive.google.com/file/d/1uUIyBz1iEc26-rpLQSYfeL1c3ATm6dya/view?usp=sharing which is named as snil_testing.csv

#### Step 3.1: Run for Notebook
- Open the notebook by typing 'pipenv run jupyter notebook' on terminal.
- Jupyter notebook will open up. Run each cell as instructed in the notebook.

#### Step 3.2: Run for py file
- Run "python main.py"


### Output:
- Predictions if two sentences are similar, not similar or neutral using roberta model.
- Cosine Similarity between two sentences using sentence_transformer.
- Cosine Similarity between two sentences using tfidf.
