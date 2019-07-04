Project Modules
-----------------------
The project consists of several python modules: mainly, we will find:

#### Embeddings

Contains classes for learning graph embeddings using various techniques, in particular:
* Laplacian Eigenmaps
* Deep Walk
* node2vec
* SDNE
* LINE

Main Class for accessing these embeddings is *EmbeddingFactory*, stored in *embedding_factory.py*.

#### Evaluations

Contains methods for evaluating the created graph embeddings, in particular:

* Visualisation
* Graph Reconstruction
* Link Prediction

#### Experiments

Contains various experiments executed to prove the functionality of the graph embeddings. Thanks to the functionality implemented in the *Run* modules, we can easily create and evaluate the embeddings in various test scenarios.

#### Graphs

Contains methods for loading and processing input graphs. Currently, the *networkx* library is used for storing graphs in memory. Unfortunatelly, this library struggles with large scale networks and thus will need to be replaced.

#### Normalization

Various attempts to normalise the weights of the input graphs into <0,1> interval.

#### Reconstruction

Methods for reconstructing the graph. Used in Graph Reconstruction and Link Prediction.
* direct reconstruction produces weights as the underlying technique provides them, i.e. does not guarantee any limitations on the generated weights.
* negsam reconstruction attempts to reconstruct the weights by applying "Negative Sampling" technique, i.e. produces a softmax of the direct weight from the module with respect to weights of a randomly chosen negative sample.
* model reconstruction is an attempt to produce a neural network which, given the embeddings of two vertices, would predict the weight of the edge between them. Note, that this is currently work in progress.

#### Run
Its main class *RunStages* is supposed to handle all functionality of this solution. Both *main.py* and all experiments are just wrappers around this class.
