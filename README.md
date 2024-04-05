# Classifying diphone audios in Julia

## Project summary

This is a proof of concept for one of the studies in the GaLa project. The aim is the achieve overall successful classification of diphones into consonant-vowel (CV) or vowel-consonant (VC) by an Recursive Neural Network (RNN) that takes individual audios (WAV files) as input, extract relevant acoustic features from them, and outputs a label 'CV' or 'VC' predicting the type of diphone that the audio is.

Early iterations of this model will involve supervised learning: the RNN will be provided with labelled audios during training. The next step will be to train the model in an unsupervised learning task, in which unlabelled audios are provided, and the model must figure out how to classifiy the CV and VC audios, base solely on their (divergent) acoustic properties. This unsupervised learning provides a stronger test, closer to what human infants encounter in experimental tasks (see Santolin et al., 2023).

The codebase of this project is inspired by the EARSHOT model by Magnuson et a. (2020).

### Audios

Audios were downloaded from the [EARSHOT supplementary materials](https://drive.google.com/file/d/1poWuCQ1_09jBSaIZJbj5KvBLl4HejdH3/view).

### Data processing

### Model structure

### Testing the model

## Reproducibility

First, install the Julia packages needed (listed in `Manifest.toml`, but see code below to install them automatically):

```julia
Pkg.instantiate()
```