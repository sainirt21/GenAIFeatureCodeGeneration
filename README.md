# GenAIFeatureCodeGeneration
The main goal is to generate the cucmber feature files and their step definitions.

So for data preparation, we are traversing all the files (.java and .feature files).

Preparing the data mappings to pass on as training data to fine-tune gpt-4 model as per our code existing code.

For data mappings, we are using the cosine similarity between step and step definition tagline.

Write now this code is as per the formatting of my code repo, but can be used for the generic code repos also after some minor adjustments.


This repository contains two Jupyter notebooks: `dataPreparation.ipynb` and `cuc_stepgeneration.ipynb`. Below are the instructions and purposes for each cell in these notebooks.

dataPreparation.ipynb

1. **Cell 1:**
    - For preparing json data for Step to Step definition function mapping

2. **Cell 2:**
    - For preparing json data for Scenario to its steps mapping

3. **Cell 3:**
    - Preparation of JSONL data from the above two json data files, which will be used for fine tuning the model.

4. **Cell 4:**
    - Preparation of text files for all the existing steps which will feeded in prompt to gpt model

5. **Cell 5:**
    - Preparation of text files for all the existing sceanrios which will feeded in prompt to gpt model


cuc_stepgeneration.ipynb

1. **Cell 1**
    - Python Script to generate scenario as well its step definition if step do not exists already. In this we are using 2 models, one is gpt-4 base model to get the scenario file using existing steps in prompt, and thens its reponse is passed to the fine tuned model for generating new steps defintions. Also we are using embedding model before gpt-4 base model to create embedding for existing steps.

2. **Cell 2:**
    -  This is IN PROGRESS. The main diference form above script in this script is including scenario files as well for prompt.
    Python Script to generate scenario as well its step definition if step do not exists already. In this we are using 2 models, one is gpt-4 base model to get the scenario file using existing steps and scenarios as well in prompt, and thens its reponse is passed to the fine tuned model for generating new steps defintions. Also we are using embedding model before gpt-4 base model to create embedding for existing steps and scenarios.
