import pandas as pd
import re
import string
import random
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from datasets import load_dataset

# run two commands and then load the below:
# 1. python -m spacy download en_core_web_sm
# 2. python -m spacy download en
nlp = spacy.load('en_core_web_sm')


# For running stem_flag="Lem" download the below:
# nltk.download('wordnet')

def download_data(dataset_name="wikipedia", dataset_id="20220301.simple"):
    output_data = load_dataset(dataset_name, dataset_id)
    return output_data


def clean_html(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'code', 'a']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


def clean_string(input_text,
                 lowercase_flag=True,
                 remove_line_breaks_flag=True,
                 normalize_punctuation_flag=True,
                 remove_stop_words_flag=False,
                 remove_numbers_flag=True,
                 stem_flag="None"):
    # Make lower
    if lowercase_flag:
        input_text = input_text.lower()

    # Remove line breaks
    if remove_line_breaks_flag:
        input_text = re.sub(r'\n', ' ', input_text)

    # Remove puncuation
    if normalize_punctuation_flag:
        # map punctuation to dot
        input_text = input_text.translate(str.maketrans(":;!?", "." * len(":;!?")))
        # map punctuation to space
        input_text = input_text.translate(str.maketrans("-|", " " * len("-|")))
        # #map punctuation to no space
        # input_text = input_text.translate(str.maketrans("","","-"))

    # Remove stop words
    if remove_stop_words_flag:
        input_text = input_text.split()
        useless_words = nltk.corpus.stopwords.words("english")
        useless_words = useless_words + ['hi', 'im']

        text_filtered = [word for word in input_text if not word in useless_words]
    else:
        text_filtered = input_text
    # print("post number removal=", text_filtered)
    # Remove numbers
    if remove_numbers_flag:
        text_filtered = re.sub('[0-9]+', '', text_filtered)
    # print("post remove_stop_words_flag=", text_filtered)
    # Stem or Lemmatize
    if stem_flag == 'Stem':
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem_flag == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem_flag == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered
    # print("text_stemmed=", text_stemmed)
    final_string = ''.join(text_stemmed)
    # Remove Multiple spaces
    final_string = re.sub(' +', ' ', final_string)
    return final_string


def get_cleaned_sentences(input_dataset):
    out_articles = []
    for i in range(len(input_dataset)):
        article = input_dataset[i]
        # print("article pre processing = ", article)
        article = clean_html(article)
        # print("article post html cleaning = ", article)
        article = clean_string(article, lowercase_flag=False, remove_line_breaks_flag=True,
                               normalize_punctuation_flag=True, remove_stop_words_flag=False, remove_numbers_flag=True,
                               stem_flag="Lem")
        # print("article post full cleaning = ", article)
        out_articles.append(article)

    return out_articles


def create_data_subset(input_dataset, num_samples=1000):
    dataset = []
    for i in range(len(input_dataset['train'])):
        article = input_dataset['train'][i]['text']
        dataset.append(article)
    data_subset = random.sample(dataset, num_samples)
    return data_subset


if __name__ == "__main__":
    random.seed(9)
    wikidata = download_data(dataset_name="wikipedia", dataset_id="20220301.simple")
    wikidata_subset = create_data_subset(input_dataset=wikidata, num_samples=1000)
    cleaned_subset = get_cleaned_sentences(input_dataset=wikidata_subset)
    filename = "outputs/cleaned_sentences.txt"
    with open(filename, 'w') as f:
        for line in cleaned_subset:
            f.write(line)
            f.write('\n')

    print(len(cleaned_subset))
