# Thesis background
This Github repository is created for the Master thesis project – Creating ontology in the drink water distribution network domain. The author is Rhea Huang and from University of Amsterdam (y1kelinblue@gmail.com). 

# Introduction
The LLM plays an important role in AI development, it offers a great convenience in the Natural Language Processing task such as text summarization. The ontology construction could be considered as time consuming, therefore using LLM for the ontology learning should be explored. This project leverages RAG for ontology learning in the drinking water domain. The output for each model is available at "Output" folder. 

Below is a flowchart visualize the logic behind the code:

![FLOWCHART3 0](https://github.com/Rhea0000/Thesis/assets/145769931/33e57c24-e540-4ebf-84c2-498b19fab30d)

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
2. Install any missing packages prompted by the IDE.
3. Replace the default file locations for `Parameters.yaml` and `Document_Input` at lines 47 and 90 with the paths to your files.
4. Input your API key for OpenAI, Hugging Face, or another compatible platform for your LLM.
5. Define your embedding models and LLM in the code.
6. Run the code either in its entirety or step by step.

## Troubleshooting

If you encounter a 500 server error, it may indicate that the LLM is unable to process your query. In this case, try the following:
- Retry the request.
- Modify the query to ensure it is within the LLM's processing capabilities.

For further assistance, please open an issue in this repository.

# Reference
@misc{OpenSourceLLMs,
  title = {OpenSourceLLMs},
  year = {2024},
  url = {\url{https://colab.research.google.com/drive/1cW01yqNxKqYHpU7uvi5UfUsNnbfxVw7R}},
  note = {Accessed: 2024-04-14}
}
