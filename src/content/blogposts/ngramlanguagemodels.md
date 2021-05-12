---
slug: "/blog/ngramlanguagemodels"
date: "2021-04-28"
title: "N-Gram Language Models"
category: "3 Natural Language Processing"
order: 1
---

### Introduction

A language model predicts a probability distribution of language.
A simple and popular modeling approach to language modeling is the N-gram language model.
The N-gram language model will be described in this post, in the context of two possible applications: text generation as well as text classification (spam detection).
This post will only be covering N-gram language models at the word level.
It should be noted that N-gram models can also be developed at the character or syllable level.

### N-gram Language Modeling

An N-gram language model is a model that predicts the probability of a sequence based on the product of each word and the preceding words in the sequence to that word. 
The N-gram language model relies on the Markov assumption, which is that a current state only depends upon a limited number of prior states.

The equations below display how to calculate the probability of a sequence given a corpus, as well as how to calculate the probability of an N-gram sequence ending in a particular word given a corpus.

$$
\begin{aligned}
    C(s) &\rightarrow \text{ Count of sequence in corpus}\\
    N &\rightarrow \text{ Size of sequence}\\
    n &\rightarrow \text{ Size of Ngram}\\
    P(w_{1:N}) &= \prod^N_{i=1} P(w_i|w_{i-n+1:i-1})\\
    &\propto \sum^N_{i=1} \log P(w_i|w_{i-n+1:i-1})\\
    P(w_i|w_{i-n+1:i-1}) &= \frac{C(w_{i-n+1:i-1}, w_i)}{C(w_{i-n+1:i-1})}\\ 
\end{aligned}
$$

The above approach to calculating sequence probabilities will have issues when dealing with words that have never been seen before. 
A general approach to dealing with unseen words is to assign some small probability to unseen events.
This approach is called smoothing. 
One particular kind of smoothing, called Laplace smoothing assigns a constant to all ngram counts.
The sequence probability calculation adjusted using Laplace smoothing is shown below:

$$
\begin{aligned}
    V &\rightarrow \text{ Size of vocabulary}\\
    P(w_i|w_{i-n+1:i-1}) &= \frac{1 + C(w_{i-n+1:i-1}, w_i)}{V + C(w_{i-n+1:i-1})}\\ 
\end{aligned}
$$

Another form of smoothing, known as Good-Turing smoothing, takes into account the frequency of words with one less number of occurrences when calculating a probability for a word.
The sequence probability calculation adjusted using Good-Turing smoothing is shown below:

$$
\begin{aligned}
    N_r &\rightarrow \text{ Count of N-grams that appear $r$ times}\\
    c_*(w_i) &= (C(w_{i-n+1:i-1}, w_i) + 1) \frac{N_{C(w_{i-n+1:i-1}, w_i) + 1}}{N_{C(w_{i-n+1:i-1}, w_i)}} \\
    P(w_i|w_{i-n+1:i-1}) &= \frac{c_*(w_i)}{\sum^{\infty}_{r=1} N_r}\\
\end{aligned}
$$

Perplexity is an internal measurment used to measure the power of a language model.
The perplexity of a test set given a language model is the inverse probability of the test set, scaled by the number of words.
The lower the perplexity, the higher the conditional probability of the word sequence.
The perplexity measure equation is displayed below:

$$
\begin{aligned}
p(W) &= \sqrt[N]{\prod^{N}_{i=1} \frac{1}{P(w_i|w_{i-n+1:i-1})} }\\
&= \left[\prod^{N}_{i=1} \frac{1}{P(w_i|w_{i-n+1:i-1})}\right]^{\frac{1}{N}}\\
&= \left[\prod^{N}_{i=1} P(w_i|w_{i-n+1:i-1})\right]^{-\frac{1}{N}}\\
&= e^{\ln  \left[\prod^{N}_{i=1} P(w_i|w_{i-n+1:i-1})\right]^{-\frac{1}{N}}}\\
&= e^{ -\frac{1}{N} \ln  \left[\prod^{N}_{i=1} P(w_i|w_{i-n+1:i-1})\right]}\\
&= e^{ -\frac{1}{N} \left[\sum^{N}_{i=1} \ln P(w_i|w_{i-n+1:i-1})\right]}\\
&= \exp\left(-\frac{1}{N} \sum^{N}_{i=1} \ln P(w_i|w_{i-n+1:i-1})\right)\\
\end{aligned}
$$

### Code

Code for an NGram language model that uses Good-Turing smoothing when calculating probabilities is displayed below: 


```python
from typing import List
import re

from tqdm import tqdm
import numpy as np


class NGramLanguageModel:


    def __init__(self, n: int, pad: str="<PAD>") -> None:
        """ Instantiate NGram language model

        Args:
            n: size of NGrams
            pad: string to use as padding
        """
        self.vocab = set([])
        self.ngrams = {}
        self.totals = {}
        self.pad = pad
        self.n = n


    def parse(self, line: str) -> List[str]:
        """ Parse string and turn it into list of tokens, removing all
            non-alphanumeric characters and splitting by space

        Args:
            line: string to be parsed

        Returns:
            list of parsed tokens
        """
        line = re.sub("[^a-z\s]", "", line.lower().strip())
        tokens = [t for t in line.split(' ') if t != '']
        return tokens


    def add(self, line: str) -> None:
        """ Add line to language model

        Args:
            line: string to be added to language model
        """
        seq = [self.pad for i in range(self.n)]
        tokens = self.parse(line)

        # Split list of tokens into NGrams and add to model
        for i in range(len(tokens)):
            seq = seq[1:] + [tokens[i]]                
            tseq = tuple(seq[:-1])

            if tseq in self.ngrams:

                if tokens[i] in self.ngrams[tseq]: 
                    self.ngrams[tseq][tokens[i]] += 1

                else: 
                    self.ngrams[tseq][tokens[i]] = 1

            else: 
                self.ngrams[tseq] = { tokens[i]: 1 }

            self.vocab.add(tokens[i])


    def calcSmoothingCounts(self) -> None:
        """ After all lines are added, this method can be called to generate
            counts required for Good-Turing smoothing
        """

        self.totalCounts = 0
        self.counts = {} 

        for ngram in tqdm(self.ngrams):
            for token in self.ngrams[ngram]:
                c = self.ngrams[ngram][token]
                self.counts[c] = self.counts[c] + 1 if c in self.counts else 1
                self.totalCounts += 1


    def prob(self, sequence: List[str]) -> float:
        """ Calculate probability of sequence being produced by language model,
            smoothed using Good-Turing smoothing

        Args: 
            sequence: sequence as a list of tokens

        Returns:
            probability of language model producing sequence                
        """

        tseq = tuple(sequence[:-1])

        c = 0
        if tseq in self.ngrams and \
           sequence[-1] in self.ngrams[tseq]:
            c += self.ngrams[tseq][sequence[-1]]           

        ncn = self.counts[c+1] if c+1 in self.counts else 0
        ncd = self.counts[c] if c in self.counts else 0
        n = ncn / ncd if ncd != 0 else 0

        cstar = (c + 1) * n
        return cstar / self.totalCounts


    def perplexity(self, dataset: List[str]) -> float:
        """ Calculate preplexity of dataset with regard to model

        Args:
            dataset: list of string sequences in testing dataset

        Returns:
            perplexity score                
        """

        perp = 0; N = 0;
        for line in dataset:

            seq = [self.pad for i in range(self.n)]
            tokens = self.parse(line)
            N += len(tokens)

            for i in range(len(tokens)):
                seq = seq[1:] + [tokens[i]]                
                prob = self.prob(seq)
                perp += np.log(prob) if prob != 0 else 0

        perp = np.exp((-1/N) * perp)
        return perp
```

### Resources

- Jurafsky, Daniel, and James H. Martin. *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition*. Pearson, 2020.
- Russell, Stuart J., et al. *Artificial Intelligence: A Modern Approach*. 3rd ed, Prentice Hall, 2010.
