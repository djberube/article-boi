from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


import langchain
from langchain.chat_models import ChatOpenAI

from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

import sys

import os
import datetime


def openai():
    return ChatOpenAI(temperature='0.8', model_name="gpt-3.5-turbo", max_tokens=2048)

def generate_text_for_description(description):
    template = """You are a master copywriter for a famous advertising firm. You love writing, and you love technology.

Write a sales letter based on the following description:

    {article_description}"""


    prompt = PromptTemplate(input_variables=["article_description"], template=template)
    llm_chain = LLMChain(llm=openai(), prompt=prompt)
    return llm_chain(description, return_only_outputs= True)['text']

def apply_suggestion_to_article(article, suggestion):
    #todo

    template = """You are a master copywriter for a famous advertising firm. You love writing, and you love technology. You submitted your first draft yesterday - and you were very happy with it. However, your editor gave you the following suggestion:

    {suggestion}

Rewrite the following sales letter based on the above suggestion:

    {article}"""

    prompt = PromptTemplate(input_variables=["article", "suggestion"], template=template)
    llm_chain = LLMChain(llm=openai(), prompt=prompt)
    return llm_chain(inputs={"article": article, "suggestion": suggestion}, return_only_outputs= True)['text']

def combine_articles(article_a, article_b):
    #todo

    template = """You are a master copywriter for a famous advertising firm. You love writing, and you love technology. You received the following sales letter from an intern, John:

    {article_a}

However, another intern, Samantha, was also assigned the same topic; she wrote this:

    {article_b}

Your assignment is to combine both articles. Remove duplicate thoughts, and reorder the text in a logical way. Blend the two articles together, using the best and most interesting parts of each. You may remove duplicate thoughts. Maintain a positive, balanced and professional tone. Because you want both interns to be happy, try to use some elements from each article. Because a varied article is more interesting, try to vary tone, sentence length, paragraph length, and word length throughout the article.

    """

    prompt = PromptTemplate(input_variables=["article_a", "article_b"], template=template)
    llm_chain = LLMChain(llm=openai(), prompt=prompt)
    return llm_chain(inputs={"article_a": article_a, "article_b": article_b}, return_only_outputs= True)['text']




def get_new_article_directory():
    # Create "out" directory if it does not exist
    if not os.path.exists("out"):
        os.makedirs("out")
    
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Find the highest index number in the subdirectories
    index = 1
    while os.path.exists(f"out/{current_date}-{str(index).zfill(3)}"):
        index += 1
    
    # Create subdirectory with current date and index number
    subdirectory = f"{current_date}-{str(index).zfill(3)}"
    os.makedirs(f"out/{subdirectory}")
    
    # Return the path to the subdirectory
    return f"out/{subdirectory}"

def log_text_to_article_directory(directory, text):

    file_names = [file for file in os.listdir(directory) if file.endswith(".txt")]

    if not file_names:
        max_file_number = 0
    else:
        max_file_number = max([int(file_name[:-4]) for file_name in file_names])

    file_name = str(max_file_number + 1).zfill(3) + ".txt"
    file_path = os.path.join(directory, file_name)
    
    print(f"logging to {file_path}")
    print(f"logging {len(text)} chars")
    with open(file_path, "w") as file:
        file.write(text)

article_directory = get_new_article_directory()

initial = open(sys.argv[1]).read()
log_text_to_article_directory(article_directory, initial)
docs = []

with open('data/suggestions') as suggestion_file:
    for suggestion in suggestion_file:
        print('processing', suggestion)
        this_step = apply_suggestion_to_article(initial, suggestion)
        log_text_to_article_directory(article_directory, this_step)
        docs.append(this_step)

chain = initial
for intermediate_doc in docs:
    chain = combine_articles(intermediate_doc, chain)
    log_text_to_article_directory(article_directory, chain)

print('Final:')
print(chain)
