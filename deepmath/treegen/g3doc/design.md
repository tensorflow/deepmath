# Generative models for structured formula generation

## Overview

We want to learn how to generate logical statements with a deep learning model.

Eventually, we'll condition the generation on some inputs. For example, after
reading in some premises and a hypothesis, we can try to predict the clauses
that an ATP might use as part of the correct steps for a proof. Alternatively,
we can try to predict an easier lemma that can be used to prove the hypothesis
we want to ultimately prove.

## Setting

We'll first consider simple arithmetic formulas where all numbers are between 0
and 9, and only operations are binary addition and subtraction.

```none
term → number | plus | minus
plus → term + term
minus → term - term
number → 0 | 1 | ... | 9
```

`deepmath/treegen/arith_make_data.py` creates arithmetic
formulas up to a certain depth which evaluate to some value.

First-order logic formulas are more complicated:

TODO(ricshin): Fill this in.

## Methods

### Sequence model

We can linearize the formula as a sequence (e.g. a tokenized version of the
standard human-readable syntax used for the formulas) and use standard sequence
modeling techniques, like a LSTM language model.

However, this has the disadvantage that generated statements may be
syntactically invalid, and the decoder needs to check that only valid statements
are generated.

### Top-down tree model

We can build a model which samples a tree top-down like this:

1.  Receive an embedding from above.
2.  Using this embedding, run it through a linear layer and a softmax layer, and
    sample the production rule to apply.
3.  Using weights specific to the sampled production rule, compute an embedding
    for each of the child nodes in the tree (or in other words, for each of the
    replacement symbols in the production rule).
4.  Repeat this recursively for each of the child nodes.

The embedding for the root node is trained, but fixed.

This model, as described above, is currently implemented in
`deepmath/treegen/arith_model.py`.

However, this model makes significant conditional independence assumptions
across sibling nodes given their parent. Specifically, each of the sibling nodes
are sampled completely independently after the parent node has been sampled.
Below we describe some approaches for allowing sibling nodes to depend on each
other.

### Tree model with left-to-right feedback

In a sequence-to-sequence model, each generated output symbol is fed back into
the network before computing the probability distribution over the next output
symbol. We can apply similar principles here.

*   Tree + sequence model

    While running the top-down tree model, feed the generated subtrees into a
    RNN and use that as side information in generation. The current output of
    the RNN can be concatenated with the tree state embedding.

    Consider generating `1 + (2 - 3)`, left to right:

    1.  Sample +, split into left and right. Add to RNN.
    2.  Right subtree: sample number (rather than + or -). Add to RNN.
    3.  Sample 1 as the number. Add to RNN.
    4.  Left subtree: sample -. Add to RNN.
    5.  Sample number. Add to RNN.
    6.  Sample 2. Add to RNN.
    7.  Sample 3. Add to RNN.

    The RNN receives the following input:

    ```none
    + ( number ( 1 ) ) ( - ( number ( 2 ) ) ( number ( 3 ) ) )
    ```

    Maybe this is too many parentheses? It may also be useful to have typed
    parentheses as is done in Grammar as Foreign Language.

    It's also unclear how this will interact with batching.

*   Down-up tree model

    Instead of using a sequential model, we could also incorporate the
    information about which nodes were actually sampled in a tree-structured
    way.

    For `1 + (2 - 3)`: ![down-up diagram](down-up_1_plus_2_minus_3.dot)

    The computations would happen in alphabetical order of the arrows.

    *   _c_, _d_: pass up information to + that a number (1) was sampled on the
        left subtree.
    *   _e_: needs to contain information about the left subtree of +.
    *   _h_, _i_: pass up information to - that a number (2) was sampled on the
        left subtree.
    *   _j_: needs to contain information about everything already generated.

    In the earlier top-down model, $$(a, e) = \text{Split}_\text{plus}(r)$$.
    Now, _e_ needs to be a function of _d_. So we get

    $$
    \begin{align*}
    a &= \text{Split}_\text{plus,left}(r) \\
    e &= \text{Split}_\text{plus,right}(r, d)
    \end{align*}
    $$

    To pass up information, we need a Join function:

    $$
    \begin{align*}
    d &= \text{Join}_\text{number}(c) \\
    &\vdots \\
    n &= \text{Join}_\text{minus}(i, m) \end{align*}
    $$

    This will significantly reduce the amount of batching which is possible.

Both of these impose left-to-right processing on the model, so that effectively
we get $$P(\text{left}, \text{right}) = P(\text{left})P(\text{right} \mid
\text{left})$$. It's unclear what the effect of this will be.

Nevertheless, we have effectively eliminated conditional independence.

### Other tree models which lack conditional independence

Is there a different way to factorize the probabilities so that there are no
conditional independences between any of the variables?

### Stochastic networks

Here are some ways to incorporate the use of random numbers along with a neural
net:

*   Neural net with random weights
*   Gaussian noise used as inputs to a neural net
*   Parameterize Gaussians with outputs from a neural net, and take samples

With such techniques, conditional independence might be fine because the neural
net can learn to map different subsets of random values to specific trees.

For example, let us consider generating formulas of depth 2 which evaluate to 0.
The possible formulas are `0`, `0 + 0`, `0 - 0`, `1 - 1`, `2 - 2`, ..., `9 - 9`.

With the top-down tree model, if we sample `-` as the top node, the left and
right children would both need to assign equal probability to numbers 0 to 9. It
has no effective way to make the left and right equal to each other.

However, if the root embedding is random, the neural net can potentially use the
randomness to pick one numerical value for both leaves when computing the
embedding to pass down to the leaves, and the leaves can generate this value
deterministically.

As an initial experiment, we can have the root embedding generated randomly and
then use the top-down tree model as before. We might be able to train this using
VAEs and GANs.

## Other problems

So far we have mostly discussed issues pertaining to getting the structure of a
formula right. Here are some other problems we'll need to tackle:

*   Reading existing formulas.

*   Dealing with a large number of possible inputs (for example, there are
    hundreds of thousands of definitions, lemmas, and theorems in Mizar).
    Methods with $$O(n)$$ complexity, such as reading each input with attention,
    might be too expensive.

*   Caching expensive computations, for example when we read in a formula and
    embed it as a vector.

*   Generating variables which will appear in the formula, and then using them
    in appropriate places.

*   How to appropriately make use of the human-selected names for definitions
    and theorems in the dataset.

*   Processing newly-generated formulas without having to train embeddings for
    them over a long period of time.

## Measurements

*   Test generated formulas to see if they meet some arbitrary criteria.
*   Log likelihood of train and test data.
*   Perplexity (should be equivalent to log likelihood normalized for length).

## Related work

### Generative adversarial networks

TODO(ricshin): Fill this in.

### Variational autoencoder

TODO(ricshin): Fill this in.

### Reasoning, attention, memory

*   Order Matters: Sequence to sequence for sets

### Application papers

*   Language to Logical Form with Neural Attention
*   [Generating Sentences from a Continuous Space]
    (http://arxiv.org/abs/1511.06349)

### Natural language syntactic parsing

Syntactic parsing is also about generating trees (although usually it only
involves attaching a tree structure to existing words). What can we learn from
how syntactic parsing is done?
