# nlp-2023-chatbot Minichallenge 2

## Goals

## Installation

Run the following commands:

```
conda create nlp-mc2 python=3.10
conda activate nlp-mc2
conda install pip
pip install -r requirements.txt
```

## Project Structure

- project_name/
  - data/
  - notebooks/
    - 0_preprocessing.ipynb
    - 1_dolly_model.ipynb
    - modeling.ipynb
  - src/
    - utils.py
  - reports/
  - README.md
  - requirements.txt


## Theoretical Background (LE5 and LE6)

## Data
For our data we took the piqa dataset from Huggingface. https://huggingface.co/datasets/piqa. The data used in the project consists of pairs of "Goal" and "Solution" sentences. Each pair represents a problem or an instruction and its corresponding solution. The "Goal" sentence provides a brief description or instruction for a given task, while the "Solution" sentence represents the desired outcome.

This dataset serves as the training, evaluation and testing data for the text completion system. The model learns from the patterns and relationships present in the goal-solution pairs to generate appropriate text completions when given a partial goal or prompt.
### Data Collection
Since our task was to create a generative Q&A model, we only took the correct answers for the respective questions. So we reated a .txt file with the "Solution" followed by the paired "Goal". These goal and solution pairs are saved in the .txt file as new lines.

### Data (Pre-)Processing
To make sense of the data, we first needed to extract and organize these pairs. The goals were identified by lines starting with "Goal:", while the corresponding solutions were marked with "Solution:". By parsing the file, we obtained tuples containing the goal and its associated solution, setting the stage for further processing.

Tokenization played a key role in breaking down the text into manageable tokens. To accomplish this, we utilized the Autotokenizer tokenizer from the transformers library. By initializing the tokenizer with the "EleutherAI/pythia-1b" model, we were able to convert the text data into a sequence of tokens. Additionally, we introduced special tokens that denoted specific elements of the prompts, such as the end of the response key and instruction key. This ensured proper formatting during training.

To create structured prompts for the language model, we employed a formatting template. This template consisted of an introduction blurb, an instruction key, a response key, and an end key. By inserting the goal and solution from each pair into this template, we obtained properly formatted prompts. This way we provided precice instructions to the model, so it could generate responses that fulfilled the given goals.

We needed to put our processed data into a dataset that the model could understand. So, we made a class called GoalSolutionDataset. It took in the tokenizer, file paths, maximum length, and formatting options. The class read our text file and grabbed the goal-solution pairs. Then, using the tokenizer, it turned the pairs into tokens. We also made sure the tokens were the right length by adding padding or cutting them if needed. Finally, we stored all this tokenized data in the dataset, ready for training, validation, and testing.

Before training our model, we had to organize our data using the DataCollatorForCompletionOnlyLM class. It made sure the labels were handled properly. Specifically, it ignored any tokens that came before the response key. This way, the model learned to generate responses that completed the instructions correctly.

## Model(s)

## Training

### Tokenizer Analysis
In order to understand the preprocessing of the data, we took a better look at the tokenizer. We used the autotokenizer for the pretrained model EleutherAI/pythia-1b. We analyzed the following cases:
- Out Of Vocabulary words
- Word contractions
- Tokenisation of special symbols
- Tokenisation of instructions
### Training Loop

## Evaluation

### Metrics

## Conclusion

## Reflection
