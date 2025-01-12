Training
==========



Data
=====

Paule requires data to be in the following format:




Training
========
For effective training you probably want to use a GPU. 


Paule consist of a number of models that are trained seperately. The models are:

- `Embedder`  Input: Log mel spectrogram, Output: Semantic embedding, is added to the target embedding

- `ForwardModel` Input: Normalized control parameters, Output: Log mel spectrogram

- `InverseModel` Input: Log mel spectrogram, Output: Normalized control parameters



