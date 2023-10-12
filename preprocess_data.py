import os
import random
import re
import tarfile
import warnings

# import urllib
import urllib.request
# from urllib.request import Request, urlopen
import nltk
import spacy
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from datasets import load_dataset
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# run two commands and then load the below:
# 1. python -m spacy download en_core_web_sm
# 2. python -m spacy download en
nlp = spacy.load('en_core_web_sm')

# For running stem_flag="Lem" download the below:
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))


# nltk.download('wordnet', download_dir=goal_dir)

def download_simplewiki_data(dataset_name="wikipedia", dataset_id="20220301.simple"):
    output_data = load_dataset(dataset_name, dataset_id)
    return output_data


def retrieve_wiki_summary_data(download_link='http://blob.thijs.ai/wiki-summary-dataset/tokenized.tar.gz',
                               download_folder='raw_dataset'):
    name = download_link.rsplit('/', 1)[-1]
    curr_wd = os.getcwd()
    goal_dir = os.path.join(curr_wd, download_folder)
    filename = os.path.join(goal_dir, name)
    if not os.path.isfile(filename):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(download_link, filename)


def unzip_dataset(filename, file_directory):
    file_path = os.path.join(file_directory, filename)
    output_filename = filename.split(".")[0] + ".txt"
    output_filepath = os.path.join(file_directory, output_filename)
    if not os.path.isfile(output_filepath):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(path=file_directory)
        tar.close()


def clean_html(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'code', 'a']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


def clean_string(input_text,
                 remove_title_flag=True,
                 lowercase_flag=True,
                 remove_line_breaks_flag=True,
                 normalize_punctuation_flag=True,
                 remove_stop_words_flag=False,
                 remove_numbers_flag=True,
                 stem_flag="None"):
    if remove_title_flag:
        input_text = " ".join(input_text.split("||| ")[1:])
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


def get_cleaned_sentences(input_dataset, remove_title_flag=True, lowercase_flag=False, remove_line_breaks_flag=True,
                          normalize_punctuation_flag=True, remove_stop_words_flag=False, remove_numbers_flag=True,
                          stem_flag="Lem"):
    out_articles = []
    for i in range(len(input_dataset)):
        article = input_dataset[i]
        # print("article pre processing = ", article)
        article = clean_html(article)
        # print("article post html cleaning = ", article)
        article = clean_string(article, remove_title_flag=remove_title_flag, lowercase_flag=lowercase_flag,
                               remove_line_breaks_flag=remove_line_breaks_flag,
                               normalize_punctuation_flag=normalize_punctuation_flag,
                               remove_stop_words_flag=remove_stop_words_flag, remove_numbers_flag=remove_numbers_flag,
                               stem_flag=stem_flag)
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


def create_data_subset_using_sampling(input_dataset="raw_dataset/tokenized.txt", num_samples=1000):
    filesize = os.path.getsize(input_dataset)  # size of the really big file
    data_subset = []
    # with open(input_dataset, mode='rb') as f:  # , encoding='rb') as f:
    with open(input_dataset, mode='r', encoding='ascii', errors='ignore') as f:
        for i in range(num_samples):
            offset = random.randrange(filesize)
            f.seek(offset)  # go to random position
            f.readline()  # discard - bound to be partial line
            random_article = f.readline()  # bingo!
            # extra to handle last/first line edge cases
            if len(random_article) == 0:  # we have hit the end
                f.seek(0)
                random_article = f.readline()  # so we'll grab the first line instead
            # some cleanup
            random_article = random_article.strip()
            # random_article = random_article.decode('ascii', 'ignore').encode("ascii")
            # random_article = str(random_article)
            data_subset.append(random_article)
    return data_subset


if __name__ == "__main__":
    random.seed(8)
    download_folder_name = 'raw_dataset'
    retrieve_wiki_summary_data(download_link='http://blob.thijs.ai/wiki-summary-dataset/tokenized.tar.gz',
                               download_folder=download_folder_name)
    print("file downloaded")
    unzip_dataset('tokenized.tar.gz', 'raw_dataset/')
    print("file unzipped")
    input_data_filename = 'tokenized.txt'
    input_data_filepath = os.path.join(download_folder_name, input_data_filename)
    wikidata_subset = create_data_subset_using_sampling(input_dataset=input_data_filepath, num_samples=1000)
    print("data subset obtained")
    # wikidata = download_simplewiki_data(dataset_name="wikipedia", dataset_id="20220301.simple")
    # wikidata_subset = create_data_subset(input_dataset=wikidata, num_samples=1000)

    # bs4 keeps coming up with some silly errors. We cna suppress these.
    warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
    cleaned_subset = get_cleaned_sentences(input_dataset=wikidata_subset,
                                           remove_title_flag=True,
                                           lowercase_flag=False, remove_line_breaks_flag=True,
                                           normalize_punctuation_flag=True, remove_stop_words_flag=False,
                                           remove_numbers_flag=False, stem_flag="None")  # You may use stem_flag = "Lem"
    print("data subset cleaned")
    for i in range(len(cleaned_subset)):
        print("i=%d, data=%s" % (i, cleaned_subset[i]))
    output_filename = "outputs/cleaned_sentences.txt"
    with open(output_filename, 'w') as f:
        for line in cleaned_subset:
            f.write(line)
            f.write('\n')
    print("data printed to file. Filename: %s" % output_filename)
    print(len(cleaned_subset))
