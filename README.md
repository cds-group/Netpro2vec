# Netpro2vec
A graph embedding technique based on probability distribution representations of graphs and skip-gram learning model

Authors: Ichcha Manipur, Maurizio Giordano, Mario Rosario Guarracino, Lucia Maddalena, Ilaria Granata - 
High Performance Computing and Networking (ICAR), Italian National Council of Research (CNR) - 
Mario Manzo - University of Naples "L'Orientale"

----------------------
Description
----------------------

Netpro2vec is a neural embedding framework, based on probability distribution representations of graphs,namedNetpro2vec. The goal is to look at basic node descriptions other than the degree, such as those induced by the TransitionMatrix and Node Distance Distribution.Netpro2vecprovides embeddings completely independent from the task and nature of the data.The framework is evaluated on synthetic and various real biomedical network datasets through a comprehensive experimentalclassification phase and is compared to well-known competitors

----------------------
Citation Details
----------------------
  
This work is the subject of the article:

Ichcha Manipur, Mario Manzo, Ilaria Granata, Maurizio Giordano*, Lucia Maddalena, andMario R. Guarracino
"Netpro2vec: a Graph Embedding Framework forBiomedical Applications".
Submitted to "IEEE/ACM TCBB JOURNAL - Special Issue on Deep Learning and Graph Embeddings for Network Biology".
 
At the current time, when using this source code please reference this work by citing the following
paper:

I. Granata, M. R. Guarracino, V. A. Kalyagin, L. Maddalena, I. Manipur, and P. M. Pardalos,
“Supervised classification of metabolic networks,” 
in 2018 IEEE Int. Conf. on Bioinformatics and Biomedicine (BIBM). 
IEEE, 2018, pp. 2688–2693.
 
Bibtex:

```
@inproceedings{granata2018supervised,
  title={Supervised classification of metabolic networks},
  author={Granata, Ilaria and Guarracino, Mario R and Kalyagin, Valery A and Maddalena, Lucia and Manipur, Ichcha and Pardalos, Panos M},
  booktitle={2018 IEEE Int. Conf. on Bioinformatics and Biomedicine (BIBM)},
  pages={2688--2693},
  year={2018},
  organization={IEEE}
}
```

----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "GPL.txt" file.

----------------------
Requirements
----------------------

To run the code the following software must be installed on your system:

1. Python 3.6 (later versions may also work)

An the following python packages:

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

In the root directory just run:

```
$ python setup.py install
```
----------------------
Running
----------------------

You can test Netpro2vec model for graph embedding by running the following python code:

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

For using Netpro2vec in you application, see the API documentation included in folder <code>html</code>.
