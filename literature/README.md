# Literature

Here we keep all relevant research articles we find.

Jawahar et al. `2020.coling-main.208.pdf` is a survey about generated-text-detection in general. Also get's into using curvature of the log probability space.

Mitchell et al. / DetectGPT `2301.11305v2.pdf` is the original paper. They're the ones who came up with the perturbation curvature idea in the first place: generated text tends to sit in a region of negative curvature of the source model's own log-probability, so perturbing it and comparing log-probabilities before/after gives away whether it's generated. Mireshghallah et al. and Bao et al. below both build on this.

Mireshghallah et al. `2305.09859v4.pdf` explains the techinique we initially wanted to use to detect generated text. It find the curvature of the joint probability of the text by inputting perturbed samples of the input text. If this curvature is high, i.e., the joint probabilty drops a lot on perturbed samples, then the text is probably generated.

Bao et al. / FastDetect `2310.05130v3` explains an alternative technique which is faster because it doesn't rely on costly perturbations. This will probably now be the we'll be using.

Schoenbach & Rosamond `Standardization.pdf` is chapter 6 (Standardization of rates and ratios) of the textbook Understanding the Fundamentals of Epidemiology. It explains direct standardization: a measure is computed within each stratum (subgroup) and then averaged with the weights of one standard population, so that groups with a different composition can be compared. Our adjusted AUC is the same, with the AUC as the measure and equal weights per subgroup.

Joshi et al. `2020.acl-main.560.pdf` ("The State and Fate of Linguistic Diversity and Inclusion in the NLP World", ACL 2020) sorts languages into 6 classes (0–5) by how much labelled and unlabelled data and tools exist for them. We group our text languages by these classes: high = class 5, upper-mid = 4, mid = 3, low = 0–2 (`resource_level` in `evaluation/groups.py`). The class of every language is listed in https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt.
