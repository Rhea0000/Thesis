from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
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

#Read the external parameter file
if __name__ == '__main__':
    with open("C:/Users/y1kel/Documents/Master IS/Thesis/Thesis code/Parameters.yaml") as stream:
        try:
            Parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("An error occurred while reading the 'Parameters.yaml' file:", exc)


#Set up the enviroment for KEYs
# OpenAI_API_KEY=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
# HUGGINGFACEHUB_API_TOKEN = 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

os.environ['OPENAI_API_KEY'] = input("Enter the key:")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = input("Enter the key:")
llm= ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.5)
# llm= HuggingFaceHub(
#     repo_id="huggingfaceh4/zephyr-7b-alpha",
#     model_kwargs={"temperature": 0.5, "max_length": Max_length,"max_new_tokens":Max_new_totkens}
# )

# Step 1 load the document
document_path = Parameters["document_path"]
loader = dl.PyPDFLoader(document_path)
content = loader.load()

# Step 2 Split & Step 3 Embeddings
#Split the document
Chunk_size = Parameters["Chunk_size"]
Chunk_overlap = Parameters["Chunk_overlap"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap) 
chunks = text_splitter.split_documents(content)

#Embedding the chunks
embeddings = Parameters["embeddings"]
vectorstore = Chroma.from_documents(chunks, embeddings)

#Define the retriever
Search_type = Parameters["Search_type"]
Search_kwargs = Parameters["Search_kwargs"]
retriever = vectorstore.as_retriever(
    search_type=Search_type,
    search_kwargs={'k': Search_kwargs} 
)

# Step 4 RAG Chain
template = Parameters["template"]
#Prompt template setup
prompt = ChatPromptTemplate.from_template(template)

#RAG Chain setup
rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Define time function
def get_amsterdam_time():
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    utc_now = datetime.datetime.utcnow()
    amsterdam_now = utc_now.astimezone(amsterdam_tz)
    return amsterdam_now.strftime("%Y-%m-%d %H:%M:%S")

#The function for re-trying the model
Max_retries=Parameters["Max_retries"]
def retrieve_with_retry(query, max_retries=Max_retries):
    retry_count = 0
    while retry_count < max_retries:
        result = rag_chain.invoke(query)
        if result:
            return result
        retry_count += 1
    print(f"Failed to retrieve after {max_retries} attempts for query: {query}")


#The function for adjusting the result format
def result_processing(result, keyword):
    try:
        if keyword in result:
            # Extract text after keyword
            result = result.split(keyword, 1)[1].strip()
            return result
        else:
            # Keyword not found, return empty dictionary
            return {}
    except json.JSONDecodeError as e:
        print(f"Error occurred: {e}")
        return {}

#The function for finding the dictionary
def find__python_dict(text):
    stack = []
    start_index = None

    for index, char in enumerate(text):
        if char == '{':
            if not stack:
                start_index = index
            stack.append('{')
        elif char == '}':
            stack.pop()
            if not stack and start_index is not None:
                try:
                    # Attempt to parse the potential dictionary
                    potential_dict = ast.literal_eval(text[start_index:index+1])
                    if isinstance(potential_dict, dict):
                        return potential_dict
                except (SyntaxError, ValueError):
                    pass
                start_index = None
    return None

#Check the condition of the model 
def model_check(result):
    if hasattr(llm, 'repo_id') and 'huggingface' in llm.repo_id:
        print("1")
        result = result_processing(result,keyword="Assistant")
    elif hasattr(llm, 'model_name') and 'gpt' in llm.model_name:
        pass
    else:
        pass
    return result

#Load the query
QueryQ1 = Parameters["QueryQ1"]
QueryQ234 = Parameters["QueryQ234"]
QueryQ5 = Parameters["QueryQ5"]
QueryQ6 = Parameters["QueryQ6"]
QueryQ7 = Parameters["QueryQ7"]

#Domain terminology prompting
ResultQ1=retrieve_with_retry(QueryQ1)
ResultQ1=model_check(ResultQ1)
ResultQ1 = find__python_dict(ResultQ1)
print(ResultQ1)
type(ResultQ1)

# Synonymy, Taxonomy, and Predication prompting
if isinstance(ResultQ1, dict):
    term_answers = {}
    for term, definition in ResultQ1.items():
        QueryQ234=QueryQ234.replace("term",term).replace("definition",definition)
        ResultQ234=retrieve_with_retry(QueryQ234)
        ResultQ234=model_check(ResultQ234)
        term_answers[term] = ResultQ234
    print(f"ResultQ234:{term_answers}")
else:
    print(f"Query1 did not return a dictionary.ResultQ1={type(ResultQ1)}")

# Parthood prompting
if ResultQ1 is not None:
    QueryQ5=QueryQ5.replace("ResultQ1",str(ResultQ1))
    ResultQ5 = retrieve_with_retry(QueryQ5)
    ResultQ5=model_check(ResultQ5)
    print(f"Result5:{ResultQ5}")
else:
    print(f"Query1 did not return an answer.")

# Ontology Schema
if ResultQ5 is not None:
    QueryQ6=QueryQ6.replace("ResultQ5",ResultQ5)
    ResultQ6 = retrieve_with_retry(QueryQ6)
    ResultQ6=model_check(ResultQ6)
    print(f"ResultQ6:{ResultQ6}")
else:
    print(f"Query5 did not return an answer.")

# Final Ontology
if ResultQ6 is not None:
    for term, dic in term_answers.items():
        QueryQ7 = QueryQ7.replace("term", term).replace("definition", dic).replace("ResultQ6", ResultQ6)
        ResultQ6 = retrieve_with_retry(QueryQ7)
        ResultQ6=model_check(ResultQ6)
    print(f"ResultQ7: {ResultQ6}")
else:
    print("Error occurred: ResultQ6 is undefined")

# Save the result
# with open("Results.ttl", "w") as file:
#     file.write(str(ResultQ6))

