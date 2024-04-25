# Rhea Huang (13019139), University of Amsterdam 
# Last update: 2024 - 04 - 14
# This code is created for the Master thesis project -- Large Language Model for Ontology Learning in the drinking water distribution network domain. This code enables fully automatic ontology creation by building the RAG Pipeline. 
# Instruction link: https://github.com/Rhea0000/Thesis

from langchain import HuggingFaceHub
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain import embeddings
from langchain import text_splitter as ts
from langchain import vectorstores as vs
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from operator import itemgetter
from requests.exceptions import HTTPError
import torch
from transformers import pipeline
from langchain import document_loaders as dl
import pandas as pd
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_relevancy,
    context_recall,
    answer_similarity,
    answer_correctness,
)
import re
import time
import json
import datetime
import pytz
import string
import os
from getpass import getpass
import ast
from langchain.document_loaders import UnstructuredPDFLoader
import yaml
import requests
import copy
import difflib

#Read the external parameter file
if __name__ == '__main__':
    with open("C:/Users/y1kel/Documents/Master IS/Thesis/Thesis code/Parameters.yaml") as stream:
        try:
            Parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("An error occurred while reading the 'Parameters.yaml' file:", exc)


#Set up the enviroment for KEYs
os.environ['OPENAI_API_KEY'] = input("Enter the key:")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = input("Enter the key:")

#Model selections
#Select the embeddings models. Embeddings create a vector representation of a piece of text. 
embeddings= OpenAIEmbeddings()
embeddings=HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

#Select the LLM models. LLM is a core component of the ontology learning. It is generating the answer of the query.
#Read the parameters of LLM before you need it in the code
Max_length  = Parameters["Max_length"] 
Max_new_tokens = Parameters["Max_new_tokens"]
Temperature = Parameters["Temperature"]

llm= ChatOpenAI(model_name="gpt-4-0125-preview", temperature=Temperature)

llm=ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=Temperature)

llm=ChatOpenAI(model_name="gpt-4-turbo-2024-04-09", temperature=Temperature)

llm= HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": Temperature, "max_length": Max_length,"max_new_tokens":Max_new_tokens}
)

llm= HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-beta",
    model_kwargs={"temperature": Temperature, "max_length": Max_length,"max_new_tokens":Max_new_tokens}
)

# Step 1 Read the document path & Load the document
document_path = r"C:\Users\y1kel\Documents\Master IS\Thesis\Dataset\Input_document.pdf"
loader = dl.PyPDFLoader(document_path)
content = loader.load()

# Step 2 Split & Step 3 Embeddings
#Read the parameters for split & Split the document
Chunk_size = Parameters["Chunk_size"]
Chunk_overlap = Parameters["Chunk_overlap"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap) 
chunks = text_splitter.split_documents(content)

#Embedding the chunks into vector stores and store them
vectorstore = Chroma.from_documents(chunks, embeddings)

#Read the parameters for retriever & Define the retriever to fetch only the most relevant pieces and pass those to RAG chain. 
Search_type = Parameters["Search_type"]
Search_kwargs = Parameters["Search_kwargs"]
retriever = vectorstore.as_retriever(
    search_type=Search_type,
    search_kwargs={'k': Search_kwargs} 
)

# Step 4 RAG Chain
#Read the template & Setup the prompt
template = Parameters["template"]
prompt = ChatPromptTemplate.from_template(template)

#RAG Chain setup
rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Define time function if you want to check the time for result
def get_amsterdam_time():
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    utc_now = datetime.datetime.utcnow()
    amsterdam_now = utc_now.astimezone(amsterdam_tz)
    return amsterdam_now.strftime("%Y-%m-%d %H:%M:%S")

# The function for re-trying retrieval operation. If it fails to retrieve a result, it retries until the maximum number of retries is reached.
# Read the parameter for the maximum number of retries
Max_retries=Parameters["Max_retries"]
def retrieve_with_retry(query, max_retries=Max_retries):
    retry_count = 0
    # Attempt to invoke the RAG chain with the query
    while retry_count < max_retries:
        result = rag_chain.invoke(query)
        # If a result is retrieved, return it
        if result:
            return result
        # Increment the retry count if no result is obtained
        retry_count += 1
    print(f"Failed to retrieve after {max_retries} attempts for query: {query}")

# Function: result_processing
# Purpose: To process the result returned by LLM and extract the relevant part of the answer. The function removes unnecessary parts of the result that follow a specific keyword. 
# Parameters:
# - result: The raw output from the LLM.
# - keyword: A specific word used to identify the start of the relevant part of the result.
def result_processing(result, keyword):
    try:
        # Check if the keyword is present in the result.
        if keyword in result:
            # If present, split the result at the keyword and return the part after it.
            result = result.split(keyword, 1)[1].strip()
            return result
        else:
            # If the keyword is not present, return the result as is.
            return result
    except Exception as e:
        # If an error occurs during processing, print an error message.
        print(f"Error occurred: {e}")


# Function: model_check 
# Purpose: To check the type of language model used and process the result accordingly. Different models may return results in different formats, and this function ensures consistency. 
# Parameters:
# - llm: The Large language model.
# - result: The raw output from the LLM.
def model_check(llm, result):
    # Check if the model is from Hugging Face and process the result.
    if hasattr(llm, 'repo_id') and 'huggingface' in llm.repo_id:
        result = result_processing(result, keyword="Assistant")
    # If the model is GPT or another type, no additional processing is done.
    elif hasattr(llm, 'model_name') and 'gpt' in llm.model_name:
        pass
    else:
        pass
    # Return the processed or unprocessed result.
    return result

# Function: find_python_dict
# Purpose: To find and return a dictionary object from a string representation of a dictionary. This function is useful for extracting dictionary objects from text that includes Python code.
# Parameters:
# - text: The string that potentially contains a dictionary.
def find_python_dict(text):
    # Initialize a stack to keep track of opening and closing braces.
    stack = []
    # Variable to store the index where a dictionary starts.
    start_index = None
    # Iterate over each character in the text.
    for index, char in enumerate(text):
        # If an opening brace is found, process it.
        if char == '{':
            # If the stack is empty, this is the start of a dictionary.
            if not stack:
                start_index = index
            # Add the opening brace to the stack.
            stack.append('{')
        # If a closing brace is found, process it.
        elif char == '}':
            # If the stack is empty, ignore this closing brace.
            if not stack:
                continue
            # Remove the last opening brace from the stack.
            stack.pop()
            if not stack and start_index is not None:
                # Attempt to evaluate the text as a dictionary.
                try:
                    potential_dict = ast.literal_eval(text[start_index:index+1])
                    if isinstance(potential_dict, dict):
                        # If successful and the result is a dictionary, return it.
                        return potential_dict
                except (SyntaxError, ValueError):
                     # If evaluation fails, reset the start index and continue.
                    pass
                start_index = None
    # If no dictionary is found, return None.
    return None

#Load the query
QueryQ1 = Parameters["QueryQ1"]
QueryQ234 = Parameters["QueryQ234"]
QueryQ234_all = Parameters["QueryQ234_all"]
QueryQ5 = Parameters["QueryQ5"]
QueryQ6 = Parameters["QueryQ6"]
QueryQ7 = Parameters["QueryQ7"]
QueryQ7_all = Parameters["QueryQ7_all"]

#Domain terminology prompting
ResultQ1=retrieve_with_retry(QueryQ1)
ResultQ1=model_check(llm,ResultQ1)
ResultQ1 =find_python_dict(ResultQ1)
print(ResultQ1)
type(ResultQ1)

# Synonymy, Taxonomy, and Predication prompting 
# Check if ResultQ1 is a dictionary to proceed with term processing 
if isinstance(ResultQ1, dict):
    term_answers = {}  # Initialize a dictionary to store term answers
    error_occurred = False  # Flag to track if an error occurs
    try:
        # Replace placeholder in QueryQ234_all with actual ResultQ1
        CopyQ234 = QueryQ234_all.replace("ResultQ1", str(ResultQ1))
        ResultQ234 = retrieve_with_retry(CopyQ234)
        ResultQ234 = model_check(llm, ResultQ234)
        print(f"ResultQ234_all_at_once:{ResultQ234}")
    except requests.HTTPError as e:
        print(f"An error occurred: {e}")
    # Iterate over each term and its definition in ResultQ1
        for term, definition in ResultQ1.items():
            try:
                CopyQ234 = copy.copy(QueryQ234).replace("term", term).replace("definition", definition)
                ResultQ234 = retrieve_with_retry(CopyQ234)
                ResultQ234 = model_check(llm, ResultQ234)
                term_answers[term] = ResultQ234 
            except requests.HTTPError as e:
                print(f"An error occurred: {e}")
                error_occurred = True 
                break # Exit the loop if an error occurs
    # If no error occurred, print the term answers
        if not error_occurred:
            print(f"ResultQ234:{term_answers}")
else:
    print(f"Query1 did not return a dictionary.ResultQ1={type(ResultQ1)}")

# Parthood prompting
if ResultQ1 is not None:
    QueryQ5=QueryQ5.replace("ResultQ1",str(ResultQ1))
    ResultQ5 = retrieve_with_retry(QueryQ5)
    ResultQ5=model_check(llm,ResultQ5)
    print(f"Result5:{ResultQ5}")
else:
    print(f"Query1 did not return an answer.")

# Ontology Schema
if ResultQ5 is not None:
    QueryQ6=QueryQ6.replace("ResultQ5",ResultQ5)
    ResultQ6 = retrieve_with_retry(QueryQ6)
    ResultQ6=model_check(llm,ResultQ6)
    print(f"ResultQ6:{ResultQ6}")
else:
    print(f"Query5 did not return an answer.")
# Make a copy of ResultQ6 for later use
ResultQ6_Copy=ResultQ6

# Final Ontology
# Check if ResultQ6 is not None and term_answers is not empty to proceed with final ontology
if ResultQ6 is not None and len(term_answers) == 0:
    final_ontology_info = {} # Initialize a dictionary to store final ontology information 
    try:
        # Replace placeholders in QueryQ7_all with actual results
        CopyQ7 = QueryQ7_all.replace("ResultQ234", ResultQ234).replace("ResultQ6", ResultQ6)
        ResultQ7 = retrieve_with_retry(CopyQ7)
        ResultQ7 = model_check(llm, ResultQ7)
        print(f"ResultQ7_all_at_once': {ResultQ7}")
    except requests.HTTPError as e:
        print(f"An error occurred: {e}")
# Check if term_answers is not empty
elif term_answers != {}:
    final_ontology_info = {} # Initialize a dictionary to store final ontology information
    # Iterate over each term and its associated dictionary in term_answers
    for term, dic in term_answers.items():
        try: 
            CopyQ7 = copy.copy(QueryQ7).replace("term", term).replace("dic", dic).replace("ResultQ6", ResultQ6)
            ResultQ6 = retrieve_with_retry(CopyQ7)
            ResultQ6 = model_check(llm, ResultQ6)
            final_ontology_info[term] = ResultQ6
        except requests.HTTPError as e:
            print(f"An error occurred: {e}")
    # Assign the processed ResultQ6 to ResultQ7
    ResultQ7 = ResultQ6
    print(f"ResultQ7_build_on_top: {ResultQ7}")
    print(f"final_ontology_info: {final_ontology_info}")
# Check if ResultQ6 is undefined
elif not ResultQ6:
    print("Error occurred: ResultQ6 is undefined")
else:
    # Print an error message for an unknown condition
    print("Error occurred: Unknown condition")


# Save the result
with open("Results.ttl", "w") as file:
    file.write(ResultQ7)

# Similarity check
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity between multiple texts
def calculate_similarity(texts):
    # Convert the text to TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity matrix
    cos_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_similarity_matrix

# Compare
responses =["""Enter your inputs"""]
# Calculate and print the cosine similarity matrix
similarity_matrix = calculate_similarity(responses)
print("Cosine similarity matrix:\n", similarity_matrix)
