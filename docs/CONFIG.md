Configuration and Parameters
-----------------------

The configurations within the project are handled by:
* .json config files
* script parameters

### .json config files
Are parameters, which are fairly stable and/or dataset-specific. Across the solution, we may find:

##### ds_config.json
Is a single configuration file which specifies entries for the datasets. Mainly, the config file specifies paths to various datasets, together with paths to the source files and parser configurations.

Note, that the name of a dataset entry used by the *main.py* script as *--dataset* parameter.

##### em_config.json
As each embedding method has various amount of parameters, we will specify them in a separate configuration file. 
While we can have one, common configuration in the root of the config directory, the solution supports dataset-specific configurations, which are situated in:
\
*/$CONFIG_ROOT/[dataset_name]/em_config.json*

##### ev_config.json
Like the embedding configurations, the evaluations are also configured by .json config files.
Likewise, those configs can either be common for the whole dataset, or dataset-specific, i.e.
\
*/$CONFIG_ROOT/[dataset_name]/ev_config.json*


##### main.py parameters
The main script has many various parameters. Some of the most important are:

* config - root of the configuration files
* output - root of the output files


* dataset - name of the dataset, i.e. key to the entry in *ds_config.json*
* init_norm - normalisation from the init_norms.py
* hide_edges - hide x% from the graph edges randomly


* embed_method - method to use, has to have entry in *em_config.json*
* embed_dim - dimension of the embeddings
* eval_method - which evaluation to use. None if not required.
* add_edges - Add x% of the graph's edges 


Also, running *--help* can give you all the available parameters with their meanings.