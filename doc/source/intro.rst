Quickstart Guide
======================================

**Netpro2vec** is released as a python package. Please contact the authors
to get the latest software release.

Installation
************

You can install **Netpro2vec** with the command:

.. code-block:: bash

	$ python setup.py install


Testing
*******

You can test the **Netpro2vec** embedding method by 
executing the ``test.py``program:

.. code-block:: bash

	$ python test.py
	     --inputpath <your-path-to-graph-dir>
	     --labelfile <your-path-to-class-label-file>
	     --labelpos <class-label-position-in-label-file>
	     --dimensions 512
	     --distributions ndd
	     --aggregators 0
	     --cutoffs 0.05
	     --extractor 1
	     --savefile <embed-filepath>

This command takes as input the directory of the graphml files and the graph class labels file. It
produces an embedding (with 512 variables) based on NDD annotation. The node annotation is parametrized by the aggregator, extractor and cut-off paramters. The resulting embeddings are saved (together with graph class labels) into the CSV file ``<embed-filepath>``.
Once the embedding is saved into a CSV file, it can be use for classification with other external tools or toolkit (like Weka).
In the same ``test.py`` script the validation is implemented, as an example, by 
a ten-fold cross-validation with a SVM classifier (with linear kernel and default paramters) implemented in the `scikit-learn <https://scikit-learn.org/>`_ python library.

.. code-block:: bash

	$ python test.py
	     --loadfile <embed-filepath>
	     --select
	     --validate

The  ``select`` parameter enables embedding reduction by recursive feature elimination 
by SVM (RFE-SVM).


Running as a module
*******************

**Netpro2vec** can be loaded as a module inside a python interpreter with the command:

.. code-block:: python

	>>> import netpro2vec


Once the module is loaded, you can generate embedding with the following python code:

.. code-block:: python

	>>> import os
	>>> path = "...your-path-to-graph-directory..."
	>>> filenames = os.listdir(path)
	>>> import igraph as ig 
	>>> graphs = [ig.load(os.path.join(path,f)) for f in filenames]
	>>> from netpro2vec.Netpro2vec import Netpro2vec
	>>> model = Netpro2vec()
	>>> medel.fit(graphs)
	>>> model.get_embedding()
	array([[ 1.6579988e-04, -1.2643806e-03, -9.3608483e-04, ...,
	         3.5708759e-03,  9.1854345e-05, -2.4944848e-05],
	       [ 1.3941363e-03, -1.4870684e-04,  2.2236386e-03, ...,
	         2.7858163e-03, -1.1076004e-03, -1.6276642e-03],
	       [-7.9958932e-04,  3.7489494e-03, -2.2576803e-03, ...,
	        -2.2035998e-03,  2.9178325e-03, -3.3222451e-03],
	       ...,
	       [ 1.9070054e-03,  2.5690219e-04, -1.7170990e-03, ...,
	        -2.1398342e-03, -1.1024768e-03, -2.9834590e-03],
	       [-3.7194900e-03,  4.5244402e-04, -6.9161621e-04, ...,
	        -3.6566083e-03,  4.5301823e-04,  2.0657710e-04],
	       [ 4.9070415e-05,  9.1010216e-04, -2.1217461e-03, ...,
	        -2.5239761e-03, -2.7091724e-03,  9.7283931e-04]], dtype=float32)
