from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
import numpy as np

INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

class GoalSolutionDataset(Dataset):
    """
    Dataset for goal-solution pairs
    
    Args:
        tokenizer: tokenizer to use
        file_path: path to file containing goal-solution pairs
        max_length: maximum length of the input sequence
        format: whether to format the data as dolly instruct prompt or not
        padding: padding strategy to use
    """
    def __init__(self, tokenizer, file_path, max_length=512, format=True, padding = "max_length"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format = format
        self.padding = padding

        self.examples = []
        with open(file_path, "r") as file:
            goal, solution = "", ""
            for line in file.readlines():
                if line.startswith("Goal:"):
                    goal = line[0:-1] # remove newline character
                elif line.startswith("Solution:"):
                    solution = line[0:-1] # remove newline character
                else:
                    goal += " " + line[0:-1] # add to goal if it's a continuation
                
                if goal and solution: # if both goal and solution are collected
                    self.examples.append((goal, solution))
                    goal, solution = "", "" # reset for next pair

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        goal, solution = self.examples[i]
        if self.format:
            data = PROMPT_NO_INPUT_FORMAT.format(instruction=goal, response=solution)
        else:
            data = goal + "\n" + solution
        if self.padding is not None:
            tokenized_data = self.tokenizer(data, truncation=True, padding=self.padding, max_length=self.max_length)
        else:
            tokenized_data = self.tokenizer(data, truncation=True, max_length=self.max_length)
        return tokenized_data