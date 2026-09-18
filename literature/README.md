# Literature

Here we keep all relevant research articles we find.

Jawahar et al. `2020.coling-main.208.pdf` is a survey about generated-text-detection in general. Also get's into using curvature of the log probability space.

Mitchell et al. / DetectGPT `2301.11305v2.pdf` is the original paper. They're the ones who came up with the perturbation curvature idea in the first place: generated text tends to sit in a region of negative curvature of the source model's own log-probability, so perturbing it and comparing log-probabilities before/after gives away whether it's generated. Mireshghallah et al. and Bao et al. below both build on this.

Mireshghallah et al. `2305.09859v4.pdf` explains the techinique we initially wanted to use to detect generated text. It find the curvature of the joint probability of the text by inputting perturbed samples of the input text. If this curvature is high, i.e., the joint probabilty drops a lot on perturbed samples, then the text is probably generated.

Bao et al. / FastDetect `2310.05130v3` explains an alternative technique which is faster because it doesn't rely on costly perturbations. This will probably now be the we'll be using.

