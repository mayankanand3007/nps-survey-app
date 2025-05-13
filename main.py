"""
    Author: Mayank Anand
    Date Created at: 29-10-2022
    Last Modified by: Mayank Anand
    Last Modified Date: 22-11-2022
    Title: List and Word Cloud of common topics where learners shared the feedback that they need more help with and where they learnt well.
"""
# Importing Libraries
from fuzzywuzzy import process
import json
import logging
import nltk
from nltk.corpus.reader import WordListCorpusReader
from nltk.util import ngrams
import os
import pandas as pd
import random
import warnings
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Global Variables
# setting corpus fetching directories to invidual variable names
CUSTOM_CORPUS_PATH = os.getcwd() + "/nltk-custom-corpus"
# fetching list of files corpus directories.
CUSTOM_CORPUS_FILES = os.listdir(CUSTOM_CORPUS_PATH)
# using Word List Corpus to initialise NLTK corpus from corpus directory and included files.
CORPUS_READER = WordListCorpusReader(CUSTOM_CORPUS_PATH, CUSTOM_CORPUS_FILES)
# Key-value pairs of Corpus.
CORPUS_READER_DICT = {technology: fileid for fileid in CORPUS_READER.fileids() for technology in CORPUS_READER.words(fileid)}
# Creating a list of unique values of technologies available in all technologies corpus files.
TECHNOLOGY_TAGGER = list(set(CORPUS_READER_DICT.keys()))
# using Unigram Tagger to tag invidual words
TAGGER = nltk.tag.UnigramTagger(model=CORPUS_READER_DICT)
# creating a boolean variable to check if the file doesnt exist.
COUNTER = False
# Making a global variable for input directory.
INPUT_DIRECTORY = "/input/"
# Making a global variable for current working directory input path.
INPUT_PATH = os.getcwd() + INPUT_DIRECTORY


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
template = Jinja2Templates(directory="static")


def get_input_files():
    """
        Description: 
            Returns a list of input files.
        Parameters: 
            None.
        Return: 
            List of input files in data_inputs directory.
    """
    list_of_filenames = []
    for file_name in os.listdir(INPUT_PATH):
        if file_name[0] != '.':
            list_of_filenames.append(file_name)
    return list_of_filenames


def sort_asc_dict_by_values(given_dictionary):
    """
        Description: 
            Sorting given dictionary by values.
        Parameters: 
            given_dictionary: dictionary having values to be sorted.
        Return: 
            dictionary sorted in ascending order.
    """
    return dict(sorted(given_dictionary.items(), key=lambda item: item[1], reverse=False))


def sort_desc_dict_by_values(given_dictionary):
    """
        Description: 
            Sorting given dictionary by values.
        Parameters: 
            given_dictionary: dictionary having values to be sorted.
        Return: 
            dictionary sorted in descinding order.
    """
    return dict(sorted(given_dictionary.items(), key=lambda item: item[1], reverse=True))


def count_list_occurences(given_list):
    """
        Description: 
            Count occurences of unique values in given list.
        Parameters: 
            given_list: given list in which occurences are to be counted.
        Return: 
            dictionary containing values in list and their count.
    """
    list_count = {}
    for element in given_list:
        if element not in list_count:
            list_count[element] = given_list.count(element)
    return list_count


def count_technologies(tech_tag_column):
    """
        Description: 
            Gets list of technology categories tagged in input column using tag_technologies().
        Parameters: 
            tech_tag_column: String converted dictionary of tagged technologies where technology category is the key and technology tagger is the value.
        Return: 
            list of technology categories in input column.
    """
    return list(json.loads(tech_tag_column).values())


def tag_technologies(input_column):
    """
        Description: 
            Fetching technologies from a Pandas Series storing multiple rows with text given by multiple learners.
        Parameters: 
            input_column: pandas series having multiple rows including text from multiple learners used to tag technology names.
        Return: 
            Dictionary of assigned technology as keys storing single/multiple technology names filtered and tagged from input column.
    """
    # changing the text formatting of all the rows to lowecase.
    input_column = input_column.lower()
    # replacing commas in between text with spaces.
    sentence = input_column.replace(","," ")
    # tokenisation of each word in every row.
    tokens = nltk.word_tokenize(sentence)
    # creating an empty list for saving matched words in whole column.
    fuzzy_tag = []
    # looping each word from tokenized text to check for technologies and filtering them 
    # using FuzzyWuzzy.
    for token in tokens:
        matched_string = process.extractOne(token, TECHNOLOGY_TAGGER, score_cutoff=92)
        if matched_string != None:
            fuzzy_tag.append(matched_string[0])
    # tokenisation of pair of two word/ bigram in every row.
    bigram = [' '.join(e) for e in ngrams(tokens, 2)]
    # looping two words at a single occurence from tokenized text to check for technologies and filtering them 
    # using FuzzyWuzzy.
    for token in bigram:
        matched_string = process.extractOne(token, TECHNOLOGY_TAGGER, score_cutoff=92)
        if matched_string != None:
            fuzzy_tag.append(matched_string[0])
    # using Unigram tagger to custom tag FuzzyWuzzy filtered technology word or words.
    bi_pos_tag = TAGGER.tag(fuzzy_tag)
    # creating a dictionary of filtered technology words in input column and matched technology name as its key.
    filtered_bi_tags = dict(filter(lambda tag: tag[1] != None, bi_pos_tag))
    return str(json.dumps(filtered_bi_tags))


def clean_survey_name(survey_row):
    wave_list = ["Wave 1", "Wave 2", "Wave 3", "Wave 4"]
    survey_row = process.extractOne(survey_row, wave_list)[0]
    return survey_row


def get_indvidual_wave_ratings(each_file_name):
    # File name extension separated to list by '.'.
    file_name_sep = each_file_name.split(".")
    # checks file extension, if it csv or excel and reads data as pandas dataframe.
    if file_name_sep[1] == 'csv':
        data = pd.read_csv(INPUT_PATH + each_file_name)
    elif (file_name_sep[1] == 'xls') or (file_name_sep[1] == 'xlsx'):
        data = pd.read_excel(INPUT_PATH + each_file_name)
    else:
        print("{each_file_name} is not in readable format.")
        # Switching COUNTER to True as file does not exist.
        COUNTER = True
        return None
    # taking required columns for wave-wise ratings.
    ratings_data = data[["Survey Wave", "How likely will you  recommend your friend or colleague this program?"]]
    # Dropping Null Values and Setting Column data-types to string.
    ratings_data.dropna(axis=0, inplace=True)
    # Updating Index after Dropping Null Values.
    ratings_data.reset_index(inplace=True, drop=True)
    # Applying Clean Survey Name Function to Rename Wave Number values.
    ratings_data["Survey Wave"] = ratings_data["Survey Wave"].apply(clean_survey_name)
    # Get Mean Value of Wave-wise ratings.
    return {"Wave-wise Ratings": ratings_data}


def merge_ratings_dfs(existing_df, fetched_df):
    """
        Description: 
            Merges two given Pandas dataframes with different row values.
        Parameters: 
            existing_df: First Pandas dataframe to be merged.
            fetched_df: Second Pandas dataframe to be merged.
        Return: 
            dictionary including merged confident and improvement columns from both the dataframes.
    """
    fetching_df_ratings = fetched_df["Wave-wise Ratings"]
    if existing_df != {}:
        existing_df_ratings = existing_df["Wave-wise Ratings"]
    else:
        existing_df_ratings = pd.DataFrame(columns=["Survey Wave", "How likely will you  recommend your friend or colleague this program?"])
    fetching_df_ratings = pd.concat([existing_df_ratings, fetching_df_ratings])
    return {"Wave-wise Ratings": fetching_df_ratings}


def get_invidual_tech_count(each_file_name):
    """
        Description: 
            Tags technologies learnt well and technologies need help with for each of given file name in file_names list.
        Parameters: 
            each_file_name: file name in data_inputs folder of the current working directory used to filter and tag technology.
        Return: 
            sel_data: New pandas dataframe holding values already in file as well as tagged technologies that 
            learners need help in and have studied well.
            counter: True if file name is not there else returns False.
    """
    # File name extension separated to list by '.'.
    file_name_sep = each_file_name.split(".")
    # checks file extension, if it csv or excel and reads data as pandas dataframe.
    if file_name_sep[1] == 'csv':
        data = pd.read_csv(INPUT_PATH + each_file_name)
    elif (file_name_sep[1] == 'xls') or (file_name_sep[1] == 'xlsx'):
        data = pd.read_excel(INPUT_PATH + each_file_name)
    else:
        print("{each_file_name} is not in readable format.")
        # Switching COUNTER to True as file does not exist.
        COUNTER = True
        return None
    # taking required colums from whole data.
    count_data = data[["Survey Wave", "Please share the topics where you feel that you have had good learning so far", 
    "Please share the topics where you are yet not confident about your learnings"]]
    # dropping Null Values and Setting Column data-types to string.
    count_data.dropna(axis=0, inplace=True)
    count_data = count_data.astype('string')
    # applying functions to fetch technology names on columns with input given by learners.
    count_data["Technologies learners learnt well"] = count_data["Please share the topics where you feel that you have had good learning so far" \
        ].apply(tag_technologies)
    count_data["Technologies learners need help with"] = count_data["Please share the topics where you are yet not confident about your learnings" \
        ].apply(tag_technologies)
    data_count = pd.DataFrame()
    data_count["Technologies learners learnt well"] = count_data["Technologies learners learnt well" \
        ].apply(count_technologies)
    data_count["Technologies learners need help with"] = count_data["Technologies learners need help with" \
        ].apply(count_technologies)
    confident_technologies = data_count["Technologies learners learnt well"].to_list()
    improvement_technologies = data_count["Technologies learners need help with"].to_list()
    confident_tech_flattened = [element for each_list in confident_technologies for element in each_list]
    improvement_tech_flattened = [element for each_list in improvement_technologies for element in each_list]
    confident_tech_count = count_list_occurences(confident_tech_flattened)
    improvement_tech_count = count_list_occurences(improvement_tech_flattened)
    # Updating Dataframe collection with file name as keys and dataframe containing 
    # fetched technology names and text input given by user with Wave number as values.
    return {"Confident Technologies": confident_tech_count, \
        "Improvement Technologies": improvement_tech_count}


def merge_tech_count_dictionaries(existing_df, fetched_df):
    """
        Description: 
            Merges two given Pandas dataframes with different row values.
        Parameters: 
            existing_df: First Pandas dataframe to be merged.
            fetched_df: Second Pandas dataframe to be merged.
        Return: 
            dictionary including merged confident and improvement columns from both the dataframes.
    """
    fetching_df_confident = fetched_df["Confident Technologies"]
    fetching_df_improvement = fetched_df["Improvement Technologies"]
    if existing_df != {}:
        existing_df_confident = existing_df["Confident Technologies"]
        existing_df_improvement = existing_df["Improvement Technologies"]
    else:
        existing_df_confident = {}
        existing_df_improvement = {}
    fetching_df_confident = { key: fetching_df_confident.get(key, 0) + existing_df_confident.get(key, 0) \
        for key in set(fetching_df_confident) | set(existing_df_confident) }
    fetching_df_improvement = { key: fetching_df_improvement.get(key, 0) + existing_df_improvement.get(key, 0) \
        for key in set(fetching_df_improvement) | set(existing_df_improvement)  }
    return {"Confident Technologies": fetching_df_confident, \
        "Improvement Technologies": fetching_df_improvement}


def get_technology_count(file_name, no_of_files):
    """
        Description: 
            Counts tagged technologies learnt well and technologies need help with for each of given file name in file_names list.
        Parameters: 
            file_name: List of file names in data_inputs folder of the current working directory.
            no_of_files: 1 if input is single file, 2 if multiple files and 3 if tagging has to be done on all the files.
        Return: 
            tech_df_tagged_collection: Dictionary of file name as key and dataframe holding values already in file as well as 
            tagged technologies that learners need help in and have studied well.
    """
    # Creating a dictionary to store file names as keys and dataframe storing values already in file as well as 
    # tagged technologies that learners need help in and have studied well.
    tech_df_count_collection = {}
    ratings_data_collection = {}
    if no_of_files == 1:
        tech_df_count_collection = get_invidual_tech_count(file_name)
        ratings_data = get_indvidual_wave_ratings(file_name)
        wave_ratings = ratings_data["Wave-wise Ratings"].groupby(by="Survey Wave").mean()
        # Getting out Wave Number as column.
        wave_ratings.reset_index(inplace=True)
        # Calling Save Bar Chart Function to save Wave-wise ratings.
        save_barchart(wave_ratings)
    else:
        for each_file_name in file_name:
            if not COUNTER:
                tech_df_count_collection = merge_tech_count_dictionaries(tech_df_count_collection, get_invidual_tech_count(each_file_name))
                ratings_data_collection = merge_ratings_dfs(ratings_data_collection, get_indvidual_wave_ratings(each_file_name))
        wave_ratings = ratings_data_collection["Wave-wise Ratings"].groupby(by="Survey Wave").mean()
        # Getting out Wave Number as column.
        wave_ratings.reset_index(inplace=True)
        # Calling Save Bar Chart Function to save Wave-wise ratings.
        save_barchart(wave_ratings)
    return tech_df_count_collection


def save_wordcloud(confident_tech_count, improvement_tech_count):
    """
        Description: 
            Creates WordCloud of Confident Technology topics and Improvement Technology topics and saves them as image in static folder.
        Parameters: 
            confident_tech_count: Pandas series containing list of common topics of confident technologies.
            improvement_tech_count: Pandas series containing list of common topics of improvement technologies.
        Return: 
            None.
    """
    plt.rcParams["savefig.directory"] = os.chdir(os.getcwd() + "/static/images/")
    c_wordcloud = WordCloud(width = 1000, height = 500, background_color ='white').generate_from_frequencies(confident_tech_count)
    plt.figure(figsize=(15,8))
    plt.imshow(c_wordcloud)
    plt.savefig("confident_tech_cloud.png")
    i_wordcloud = WordCloud(width = 1000, height = 500, background_color ='white').generate_from_frequencies(improvement_tech_count)
    plt.figure(figsize=(15,8))
    plt.imshow(i_wordcloud)
    plt.savefig("improvement_tech_cloud.png")
    os.chdir("../../")


def save_barchart(wave_wise_ratings):
    """
        Description: 
            Creates Bar Chart of Wave-wise Ratings and saves them as image in static folder.
        Parameters: 
            wave_wise_ratings: Pandas DataFrame containing list of survey wave number and their mean value.
        Return: 
            None.
    """
    plt.rcParams["savefig.directory"] = os.chdir(os.getcwd() + "/static/images/")
    color_comb = ['red', 'yellow', 'cyan', 'blue', 'orange', 'lime', 'darkviolet', 'crimson', 'gold', 'silver', 'orangered']
    random.shuffle(color_comb)
    color_sel = []
    for index in range(4):
        color_sel.append(color_comb[index])
    plt.figure(figsize=(10,6))
    plt.bar(wave_wise_ratings["Survey Wave"], \
        wave_wise_ratings["How likely will you  recommend your friend or colleague this program?"], color=color_sel)
    plt.xlabel('Survey Wave')
    plt.ylabel('Average')
    plt.savefig("wave_rating_bar.png")
    os.chdir("../../")


@app.get("/", response_class=HTMLResponse)
def index(req: Request):
    return template.TemplateResponse("./html/index.html", {"request": req, "files": get_input_files()})


@app.post("/submit")
async def submit_form(request: Request, filename: list = Form(...)):
    common_topics = get_technology_count(filename[0], 1) if len(filename)==1 else get_technology_count(filename, 2)
    save_wordcloud(common_topics["Confident Technologies"], common_topics["Improvement Technologies"])
    return template.TemplateResponse("./html/result.html", {"request": request, \
        "impr_topics": sort_desc_dict_by_values(common_topics["Improvement Technologies"]), \
        "conf_topics": sort_desc_dict_by_values(common_topics["Confident Technologies"])})

# Driver code.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)