# Netpro2vec
A graph embedding technique based on probability distribution representations of graphs and skip-gram learning model.

Authors: Ichcha Manipur, Maurizio Giordano, Lucia Maddalena, Ilaria Granata - 
High Performance Computing and Networking (ICAR), Italian National Council of Research (CNR) - 
Mario Manzo - University of Naples "L'Orientale"
Mario Rosario Guarracino - University of Cassino and Southern Lazio

----------------------
Description
----------------------

Netpro2vec is a neural embedding framework, based on probability distribution representations of graphs. The goal is to look at node descriptions, such as those induced by the Transition Matrix and Node Distance Distribution. Netpro2vec provides embeddings completely independent from the task and nature of the data. The framework is evaluated on synthetic and various real biomedical network datasets through a comprehensive experimental classification phase and is compared to well-known competitors.

----------------------
Citation Details
----------------------
  
If you use the source code in your work please reference this work by citing the following paper:

I. Manipur, M. Manzo, I. Granata, M. Giordano, L. Maddalena and M. R. Guarracino, "Netpro2vec: A Graph Embedding Framework for Biomedical Applications," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 19, no. 2, pp. 729-740, 1 March-April 2022, doi: 10.1109/TCBB.2021.3078089.

Bibtex:

```
@ARTICLE{Manipur2021Netpro2vec,
  author={Manipur, Ichcha and Manzo, Mario and Granata, Ilaria and Giordano, Maurizio and Maddalena, Lucia and Guarracino, Mario R.},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics}, 
  title={Netpro2vec: A Graph Embedding Framework for Biomedical Applications}, 
  year={2022},
  volume={19},
  number={2},
  pages={729-740},
  doi={10.1109/TCBB.2021.3078089}}
```

----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "License.txt" file.

----------------------
Requirements
----------------------

To run the code the following software must be installed on your system:

1. Python 3.6 (later versions may also work)

And the following python packages:

```
tqdm>=4.46.1
pandas>=1.0.2
numpy>=1.16.2
gensim>=3.8.3
scipy>=1.4.1
joblib>=0.14.1
python_igraph>=0.8.2
```

----------------------
Installation
----------------------

In the root directory run:

```
$ python setup.py install
```
----------------------
Running
----------------------

You can test the Netpro2vec model for graph embedding by running the following python code:

```
>>> import os
>>> path = "data/Mutag/graphml"
>>> filenames = os.listdir(path)
>>> import igraph as ig 
>>> graphs = [ig.load(os.path.join(path,f)) for f in filenames]
>>> from netpro2vec.Netpro2vec import Netpro2vec
>>> model = Netpro2vec()
>>> model.fit(graphs)
>>> model.get_embedding()
array([[ 2.3669314e-03, -1.6807126e-03,  1.3935004e-04, ...,
         1.1605929e-03,  3.8243207e-04,  3.3924100e-03],
       [ 2.6130748e-03,  1.4776569e-03,  7.2231720e-05, ...,
        -2.7432586e-03,  2.2828898e-03,  1.6866124e-03],
       [-9.5788226e-04, -1.7322834e-03,  2.3791294e-03, ...,
        -2.7187262e-03, -1.7086907e-03,  1.4063254e-03],
       ...,
       [ 2.6994976e-03,  3.7764042e-04,  8.7952241e-04, ...,
        -3.5347182e-03, -4.4570959e-04,  2.0428676e-04],
       [ 3.1036076e-03,  2.0614895e-03, -2.9027397e-03, ...,
         3.6049024e-03, -2.0037764e-03, -6.2212220e-04],
       [-3.3860563e-03, -2.9692445e-03,  1.3172977e-03, ...,
         1.7665974e-03,  8.7682210e-04,  1.6081571e-03]], dtype=float32)
```

For using Netpro2vec in you application, see the API documentation included in the folder <code>html</code>.
