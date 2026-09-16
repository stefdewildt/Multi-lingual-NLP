# Literature

Here we keep all relevant research articles we find.

`2305.09859v4` explains the techinique we initially wanted to use to detect generated text. It find the curvature of the joint probability of the text by inputting perturbed samples of the input text. If this curvature is high, i.e., the joint probabilty drops a lot on perturbed samples, then the text is probably generated.

`2310.05130v3` explains an alternative technique which is faster because it doesn't rely on costly perturbations. This will probably now be the we'll be using.