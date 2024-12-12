# GenAIFeatureCodeGeneration
The main goal is to generate the cucmber feature files and their step definitions.

So for data preparation, we are traversing all the files (.java and .feature files).

Preparing the data mappings to pass on as training data to fine-tune gpt-4 model as per our code existing code.

For data mappings, we are using the cosine similarity between step and step definition tagline.

Write now this code is as per the formatting of my code repo, but can be used for the generic code repos also after some minor adjustments.
