# Perturbation 

This folder will hold code and files related to perturbing pieces of text. We need a way of perturbing pieces of text, since this is required for calculating the curvature (i.e. local optimality) of a piece of text for some model. 

```python 

def perturb(str) -> str:
    """ 
    Takes a string and output's a string that 
    is still close to the original string, but
    still somewhat different
    """
    pass

```

We'll be needing a few different methods on how to do this, so that we can eliminate any bias introduced by the perturbation method when we do research on performance in a multilingual.

In [Mireshghallah et al.](literature/2305.09859v4.pdf) they use the following method: randomly mask short spans of the target sequence, then use a generative text model to regenerate those spans, creating perturbed neighbor sequences. They check the models T5-Small, T5-Large and T5-3B, the masking peercentages 90%, 50%, 15%, 2% and 1%, and always use a span length of 2 tokens. 

We could do something similar, I think trying new perturbation methods should be out of scope for our research, let's focus on the language part. But we still need multiple methods or variants of the same method, to eliminate bias. I think the variants proposed in [Mireshghallah et al.](literature/2305.09859v4.pdf) should do the job. 

