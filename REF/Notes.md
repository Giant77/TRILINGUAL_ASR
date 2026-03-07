# Base model

- Wave2Vec2
- three sub-models: a pronunciation model, an acoustic model and a language model
  - **The pronunciation model** takes the pronunciation variations of words in a dictionary.
  - **the acoustic model** is a statistical model of acoustic features for sub-word units (for example, phonemes).
  - **The language model** enables the ASR system to reduce the search space by using the conditional probabilities of subsequent words with the observed word sequences.

# Train/Test dataset

- ID
- EN
- AR

# Data Prep

- mono-lang data
- sintesis code-switching data(?)

# Workflow

1. Prep Dataset
2. Train multi-lang(3) model (model A)
3. Finetune to code-switching
   1. Comparison to model A
   2. Comparison to [ID-EN Code-Switching ASR](https://arxiv.org/abs/2507.07741)
   3. Comparison to WHISPER (by GOOGLE)
