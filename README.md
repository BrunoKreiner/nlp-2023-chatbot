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
For our data, we took the Piqa dataset from Huggingface. https://huggingface.co/datasets/piqa. The data used in the project consists of pairs of "Goal" and "Solution" sentences. Each pair represents a problem or instruction and its corresponding solution. The "Goal" sentence provides a brief description or instruction for a given task, while the "Solution" sentence represents the desired outcome.

This dataset serves as the training, evaluation, and testing data for the text completion system. The model learns from the patterns and relationships present in the goal-solution pairs to generate appropriate text completions when given a partial goal or prompt.
### Data Collection
Since our task was to create a generative Q&A model, we only took the correct answers for the respective questions. So we created a .txt file with the "Solution" followed by the paired "Goal". These goal and solution pairs are saved in the .txt file as new lines.

### Data (Pre-)Processing
To make sense of the data, we first needed to extract and organize these pairs. The goals were identified by lines starting with "Goal:", while the corresponding solutions were marked with "Solution:". By parsing the file, we obtained tuples containing the goal and its associated solution, setting the stage for further processing.

We used tokenization to break down the text into manageable tokens. To accomplish this, we utilized the Autotokenizer tokenizer from the transformers library. By initializing the tokenizer with the "EleutherAI/pythia-1b" model, we were able to convert the text data into a sequence of tokens. Additionally, we introduced special tokens that denoted specific elements of the prompts, such as the end of the response key and instruction key. This ensured proper formatting during training.

To create structured prompts for the language model, we employed a formatting template. This template consisted of an introduction blurb, an instruction key ("### Instruction:"), a response key ("### Response:"), and an end key ("### End"). By inserting the goal and solution from each pair into this template, we obtained properly formatted prompts. This way we provided precise instructions to the model, so it could generate responses that fulfilled the given goals. This procedure was taken from the Dolly source code (LINK)[https://github.com/databrickslabs/dolly/blob/master/training/consts.py] where a model called Dolly was fine-tuned using the same model we chose for fine-tuning on our data which will be discussed in the next chapters.

We needed to put our processed data into a dataset that the model could understand. So, we made a class called GoalSolutionDataset extending the PyTorch dataset class. It takes in the tokenizer, file paths, maximum length, and formatting options. The class reads our text file and grabs the goal-solution pairs. Then, using the tokenizer, it turned the pairs into tokens. We also made sure the tokens were the right length by adding padding or max-length configurations to them when needed. Finally, we stored all this tokenized data in the dataset, ready for training, validation, and testing. 

Before training our model, we had to organize our data retrieved by the dataset class using a data collater. This data collator was also taken directly from the Dolly source code where it's called "DataCollatorForCompletionOnlyLM". It made sure the labels were handled properly. Specifically, it transforms the labels of any token that comes before the response key so that the PyTorch loss isn't applied to them. 

## Model(s)

As for the models used, the (Pythia 1b model)[https://huggingface.co/EleutherAI/pythia-1b] was chosen because training with the (Dolly 3b v2)[https://huggingface.co/databricks/dolly-v2-3b] and (Pythia 2.8B)[https://huggingface.co/EleutherAI/pythia-2.8b] resulted in an out of memory error. We tried using a lower batch size, gradient accumulation (up to 64), and gradient checkpointing to reduce the memory needed for training (from (this)[https://huggingface.co/docs/transformers/perf_train_gpu_one] link), but it didn't work on a single RTX 3090 with close to 24GB memory. For that reason, the smaller model was used. 

Pythia 1B is part of the Pythia suite, a collection of 16 large language models (LLMs) introduced in a research paper by EleutherAI. The Pythia 1B model, as the name suggests, is a model with 1 billion parameters. It is a decoder-only autoregressive language model (next token predictor). The model is trained on public data that EleutherAI calls "The Pile". 

On the HuggingFace page of the Pythia model, it says that it is not intended for deployment and cannot be used for human-facing interactions due to the potential of generating harmful or offensive text. It is English language only. It was not finetuned for chatting and won't behave similarly to ChatGPT since ChatGPT was finetuned using methods such as Reinforcement Learning from Human Feedback (RLHF) to better follow human instructions.  

## Training

For training, the tokenizer of Pythia-1b and the model itself were loaded using HugginFace's transformer library and its Auto** classes (AutoModelForCausalLM, AutoTokenizer) that automatically download the correct configuration from the HuggingFace hub. This is followed by the work described in the Data (Pre)Processing chapter. Then the HuggingFace trainer is setup with the following parameters:

    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=False,
    bf16=True, # https://www.cerebras.net/machine-learning/to-bfloat-or-not-to-bfloat-that-is-the-question/
    learning_rate=lr,
    logging_dir='./logs/runs',
    logging_strategy="steps",
    logging_steps=10, #dolly defaults
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb",
    remove_unused_columns=False,
    warmup_steps=0,

These were largely copied from the dolly training repository (LINK)[].

### Training Results

The training history can be seen here: https://api.wandb.ai/links/22hs_i4ds20/2q7iv0it

The evaluation loss goes down for the first 4000 training steps then it goes up in multiple big steps. The model saved by the trainer is the 3600 steps checkpoint which is used for further evaluation. The evaluation loss is pretty high compared to the training loss which gradually goes down. Therefore the model overfits pretty hard. 

### Tokenizer Analysis

In order to understand the preprocessing of the data, we took a better look at the tokenizer. We used the AutoTokenizer for the pre-trained model EleutherAI/pythia-1b. We analyzed the following cases:
- Out Of Vocabulary words
- Word contractions
- Tokenization of special symbols
- Tokenization of instructions

### Metrics

To test a language model on the task we can define multiple different metrics. Here is a list of them:

- Exact Match Percentage or BLEU
  - How many answers are the exact match? How similar are words? How much is the n-gram precision (BLEU) with n = 1 to 4?
- Grammatical Error Detection
  - Use grammar check libraries for that.
- Measure the diversity of answers
- Perplexity
  - How well a language model predicts a sample by looking at the predicted probabilities for each word in the test set.
- Ask GPT3.5 to rate the answer of a model. (Vicuna style https://lmsys.org/blog/2023-03-30-vicuna/)

## Evaluation

For the evaluation, we looked at how the Vicuna team did it to evaluate their Llama-based fine-tuned model Vicuna. They used the OpenAI API to prompt GPT 3.5 and ask it to rate and compare model answers given a question. The source code for prompting the GPT 3.5 model can be seen in (their repository)[https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl]. 

The code for our own evaluation can be seen in (99_competition.ipynb)[./99_competition.ipynb]. First, we load the fine-tuned model (lowest evaluation loss checkpoint) and the base model Pythia-1b which was not fine-tuned. Then we load the validation dataset and format it according to the models' needs. By adding "format = False" to the dataset, we ignore the Dolly formatting of the data so that the non-fine-tuned model can understand the input. Following that, we loop over 200 examples of the validation dataset. For each sample and each model, we delete the solution to the answers from the data. For the fine-tuned model this means deleting text after the response key ("### Response:\n"). For the non-fine-tuned model, this means simply deleting everything after the "Solution:" token. After that, the responses of the model over 200 examples are stored in a data frame with the original question.

The answers for each model are cleaned again to exclude the original question. Gpt 3.5 is then prompted using the Vicuna judge source code to retrieve scores for each model's answer which can then be used to rank the models. The prompt to GPT 3.5 using the OpenAI API. The prompt for GPT 3.5 looks like this:

    prompt = [
        {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
        {"role": "user", "content": f"[Question]\n{row['question_cleaned']}\n\n[The Start of Assistant 1's Answer]\n{row['response1_cleaned']}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{row['response2_cleaned']}\n\n[The End of Assistant 2's Answer]\n\n[System]\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\n\n"}
    ]

The important task to GPT 3.5 is the following: "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."

After 185 API calls to GPT 3.5, a "ServiceUnavailableError: The server is overloaded or not ready yet." was thrown. This might be due to the loop and many subsequent calls to the API. The process took 12 minutes and cost 0.16 dollars.

### Gpt 3.5 Scores

The non-fine-tuned model won 128 times while the fine-tuned model won 57 times. The average score for the non-fine-tuned model was 6.76 while the fine-tuned model scored 5.04 on average. This might be due to several issues. Let's discuss them here by looking at individual samples:

The maximum score given to the non-fine-tuned model was "9". These are some max scorer examples:

```
Goal: How to sleep in proper posture?


A:

The answer is that you can't.
The reason is that the body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing.
The body is a living thing, and it is constantly changing
```

```
Goal: How to clean blinds without tearing them up


1.  Use a vacuum cleaner to clean the blinds.
2.  Use a vacuum cleaner to clean the blinds.
3.  Use a vacuum cleaner to clean the blinds.
4.  Use a vacuum cleaner to clean the blinds.
5.  Use a vacuum cleaner to clean the blinds.
6.  Use a vacuum cleaner to clean the blinds.
7.  Use a vacuum cleaner to clean the blinds.
8.  Use a vacuum cleaner to clean the blinds.
```

As we can see, the non-fine-tuned model is already okay at giving a short answer to the question but it keeps repeating its answer and doesn't provide much more details. This is true for even the best answers according to GPT 3.5 ratings. Let's look at the answers of the fine-tuned model:

```
Goal: How to sleep in proper posture?

Stand straight and tall, hold your arms straight and your hands at your sides, then bend your arms and legs to the side, then straighten your arms and legs
```

```
Goal: How to clean blinds without tearing them up

Use a clean rag and a clean cloth.
```

These answers got a score of 3 and 4 by the GPT 3.5 model. They seem to be lacking logical reasoning. For these examples, the fine-tuned model regressed. But it's better at setting the end token and finishing its answer. This is something that it has learned from the formatted inputs by the custom dataset. Let's look at the GPT 3.5 evaluation given these responses:

```
First question: 
'9 3

Assistant 1 received a score of 9 for its helpfulness, relevance, accuracy, and level of details. The response, although short, explained the reason why it is impossible to sleep in proper posture due to the constant change in the body. It provided a clear and concise answer to the question.


Assistant 2, on the other hand, received a score of 3 for its response. The answer was not relevant to the question as it provided instructions on how to stand'
```

```
Second question:
'9 4

Assistant 1 provided a list of suggestions that were all the same, which is not very helpful or relevant to the question. However, the suggestion of using a vacuum cleaner to clean the blinds could be accurate and detailed if more information was provided on how to use the vacuum cleaner for this task. 

Assistant 2's response is concise and straightforward, but lacks detail and specificity. It is not clear what type of cloth or rag should be used, or how to effectively clean the"
```

For simplicity reasons, the call to GPT 3.5 was configured to end at 100 tokens. Still, it provides some info on how the big model ranks smaller models. The first two numbers are the scores. The first score pertains to Assistant 1's answer which is the non fine tuned model. Gpt 3.5 picks up on the repetition of answers for the non-fine-tuned model but doesn't penalize it that much. It also doesn't penalize the lack of detail in the first model's response and looking at the evaluation of the fine-tuned model's response, it focuses on the content and whether it's logical or not. We can see that the non-fine-tuned model has great scores but the scores of the fine-tuned model are significantly lower. We might assume that the fine-tuned model is only worse by a lower amount.

Let's look at the max scorers of the fine-tuned model. Here is the goal + answer of two fine-tuned model answers that got a 9/10. Apparently, 10/10 doesn't exist for both models:

```
Goal: To allow the oil for the turkey fryer to reach the correct cooking temp

Place the oil in a large pot and bring it to the desired temperature.
```

```
Goal: how to make smoky pretzel mix

Melt 1/2 cup pretzels in a skillet over medium heat. Add 1/4 cup each chopped dill weed and parsley and 1/4 cup each chopped dill weed and parsley. Cook and stir until the pretzels are crispy.
```

Here we see that the model's output makes sense to some degree. The GPT 3.5 model apparently didn't pick up on the weirdness of the second goal-answer pair where the model says one needs to melt half a cup of pretzels in a skillet. On second thought it might make sense since a "pretzel mix" might refer to small pretzel-shaped salty snacks. Let's also look at the non-fine-tuned model outputs and the GPT 3.5 judge's response:

```
Goal: To allow the oil for the turkey fryer to reach the correct cooking temp

The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the correct temperature. The oil needs to be at the'

7 9

Assistant 1's response has a repetitive and unhelpful nature, which can be frustrating for the user. It lacks relevance and accuracy, as it does not provide any useful information or instructions on how to reach the correct cooking temperature for the oil. Therefore, it receives a lower score of 7.

Assistant 2's response is relevant and accurate, providing a clear and concise instruction on how to achieve the desired temperature for the oil. It also provides a reasonable level of detail
```

```
Goal: how to make smoky pretzel mix

1.  Add the pretzel mix to the water and let it sit for about 5 minutes.
2.  Add the water to the smoky pretzel mix and let it sit for about 5 minutes.
3.  Add the water to the smoky pretzel mix and let it sit for about 5 minutes.
4.  Add the water to the smoky pretzel mix and let it sit for about 5 minutes.
5.  Add the water to the smoky pretzel mix and let it sit for about 5 minutes.'
 

8 9

Assistant 1 provided a response that did not seem relevant or accurate, as it simply repeated the same step multiple times without providing any specific instructions or details on how to make smoky pretzel mix. However, Assistant 2 provided a clear and detailed set of instructions on how to make the mix, including specific measurements and cooking steps. The only potential issue is that the recipe did not mention anything about making the mix smoky, which may be a drawback for some users. Overall
```

Interesting to see is that the non-fine-tuned model gets a fairly high score when the fine-tuned model is a high scorer. Also, the non-fine-tuned model repeats the answers again. 

## Conclusion + Plans

All in all, the fine-tuned model sets the stop token better than the baseline non-fine-tuned model. While the non-fine-tuned model is better overall according to GPT 3.5 evaluation. The fine-tuned model lacks logic more in general than the non-fine-tuned model. We assume that the fine-tuned model focused on the input format while learning. To actually be a better Q&A bot for this specific data, we need more data for training (we have around 16k training samples). The data volume is not too bad but the dataset is very broad. The prompts and questions are very vague. We wouldn't recommend the dataset for training but only for benchmarking which is its intended use anyway.

We can also note that GPU memory is a huge bottleneck. The Pythia models available on HuggingFace were trained on large clusters of GPUs with lots of memory. This is discussed in the (Pythia Paper)[https://arxiv.org/pdf/2304.01373.pdf]. During training, the max-length of the input to the model is reached a couple of times due to the input formatting and length of the goal-solution pairs. This might limit training as well.


## Reflection

