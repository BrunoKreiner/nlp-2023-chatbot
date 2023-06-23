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
