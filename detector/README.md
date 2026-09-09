# Detector Model (using Curvature)

This folder will hold code and files related to the detector model, which takes a piece of text and answers whether the text is human written or not. A simplified example, where details like tokenization are left out:

```python

def detect(
    text: str,
    dm: DetectorModel,
    pm: PerturbationModel,
    n_perturbations: int
) -> bool:
    perturbed_texts: list[str] = pm.perturb(text)
    original_log_probability: float = dm.log_probability(text)
    perturbed_log_probabilities: list[float] = [dm.log_probability(t) for t in perturbed_texts]
    curvature = original_log_probability - sum(perturbed_log_probabilities) / n_perturbations
    return curavture > curvature_threshold:

```


