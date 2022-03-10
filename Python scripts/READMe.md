# File Description 

### csv and data files

The .csv and .data files in this folder are the datasets that were used in our experiments. The **Preprocess_data.py** file contains their respective hyperlinks.

### Preprocess_data.py

This python file contains the pre-processing steps for each dataset. 

### FRU.py

This python file contains the following functions:

(1) `create_regions()`: this class object creates the membership values to the three fuzzy-rough regions.
(2) `FRU_experimental_setup()`: this class comprises of several functions. The function `initialize_objects()`initializes the crisp membership values of each instance to the fuzzy sets X (each decision class) and divides the datasets into chunks (based on instance indices) that will be fed into the machine's processors to speed up the calculation proceess. The function `fr_regions()` parallelizes the execution of `create_regions()` across the multiple input values (chunks), distributing the input data across processes (data parallelism). Finally, it reassembles the membership values of all chunks into one object. `uncertainty` calculates our proposed fuzzy-rough uncertainty measure. 

### Explicit_bias_experimental_setup.py

This python file contains the main script where users can specify the dataset of choice as well as the distance function, the implicator and conjunction and the smoothing parameter. 

### To upload - notebook that creates the synthetic dataset
