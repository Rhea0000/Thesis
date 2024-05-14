# Thesis background
This Github repository is created for the Master thesis project – Creating ontology in the drink water distribution network domain. The author is Rhea Huang and from University of Amsterdam (y1kelinblue@gmail.com). 

# Introduction
The advanced capabilities of Large Language Models (LLMs) have proven beneficial in various domains, significantly improving the efficiency of text processing and text generation. However, there is a lack of research into their application in the field of ontology learning. Currently, majority of ontologies are created manually, which is time consuming and labour intensive.
Therefore, this project focuses on the use of LLMs for ontology learning, bridging the gap by using manual ontology creation method as a basis to facilitate automatic ontology creation. The project uses the BASIC Retrieval Augmented Generation (RAG) to avoid the limitation of hallucination and has experimented with four LLMs: (gpt-4-0125-preview (gpt-4), gpt-3.5-turbo-0125 (gpt-3-turbo), gpt-4-turbo-2024-04-09 (gpt-4-turbo), huggingfaceh4/zephyr-7b-beta (7b-beta). The code file contains all the code to reproduce the result and the output for each model is available at "Output" folder. 

Below are flowcharts visualize the logic behind the RAG and query pipeline:

RAG:

![Flowchart-rag](https://github.com/Rhea0000/Thesis/assets/145769931/06890aec-822c-4c1b-b449-822c08a6cac0)


Pipeline:

![flowchart-pipeline](https://github.com/Rhea0000/Thesis/assets/145769931/51fba993-8a2b-4347-a7e4-2df38d7cf966)



# Instructions
The file "Python code" contains all the information for the reuse. 
The `Thesis_code.py` contains all the core code for creating ontology using Python.
The input document is named `Document_Input`. 
The file `Parameters.yaml` encompass all adjustable parameters in the code. 

## Prerequisites

- Python 3.8 or higher
- Visual Studio Code

## Installation

Before running the code, ensure you have the following files:
- `Parameters.yaml`
- `Document_Input`
- `Thesis_code.py`

You do not need to execute `Parameters.yaml` and `Document_Input` unless you want to change the parameters and input, they are required for the `Thesis_code.py`.

## Configuration

All default parameters are stored in `Parameters.yaml`. If you need to adjust queries or inputs, modify them in this file.

## Execution

To run the code:
1. Open `Thesis_code.py` in Visual Studio Code.
2. Install any missing packages if necessary.
3. Replace the default file locations for `Parameters.yaml` and `Document_Input` with the paths to your files.
4. Input your API key for OpenAI, Hugging Face, or another compatible platform for your LLM.
5. Define your embedding models and LLM in the code.
6. Run the code either in its entirety or step by step.

## Troubleshooting

If you encounter a 500 server error, it may indicate that the LLM is unable to process your query. In this case, try the following:
- Retry the request.
- Modify the query to ensure it is within the LLM's processing capabilities.

For further assistance, please open an issue in this repository.

# Reference
(n.d.). Quickstart. LangChain. https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
