# Generated Text Detection in a Multilingual Setting using Curvature Methods

This repo holds code and files related to the MiniProject for our university course "Multilingual Natural Language Processing", for Msc AI.  

Our main goals of the project is to try to detect AI-generated texts, using the curvative method described in [Mireshghallah et al.](literature/2305.09859v4.pdf). But we differ in the fact that we will not be testing whether the size or architecture of the model matter (like Mireshghallah et al.), but we will test whether language matters.

## Research Questions

I think our main research question should be something like:

* How does the detector's performance relate to the language of the input, and the languages the detector was trained on?
    - Do detectors still work if they were'nt trained on the input text's language.
    - Does the performance decrease when the input text's language was less prominent in the training data.
    - Is the detector performance the best for the language that the model was trained on the most (english probably).
    - Etc.

And we could also re-verify other findings:

* Do the other findings in [Mireshghallah et al.](literature/2305.09859v4.pdf) hold in a multilingual setting?
    - Do the smaller detector models still perform better than the bigger ones.
    - Is it still beneficial if the detector model has a similar architecture as the model that generated the fake text?
    - Is it still beneficial if the detector model was trained on a similar datasets as the model that generated the fake text?
    - Etc.

## Datasets

We will be making use of the MULTITuDE dataset containing the following fields per sample

* 'text' - a text sample
* 'label' - 0 for human-written text, 1 for machine-generated text,
* 'multi_label' - a string representing a large language model that generated the text or the string "human" representing a human-written text,
* 'split' - a string identifying train or test split of the dataset for the purpose of training and evaluation respectively,
* 'language' - the ISO 639-1 language code identifying the language of the given text,
* 'length' - word count of the given text,
* 'source' - a string identifying the source dataset / news medium of the given text.

For more detail check out the datasets [readme](./datasets/MULTITuDE/README.md).