import numpy as np
import igraph as ig
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from netwld2v.WeisfeilerLehman import WeisfeilerLehman
from netpro2vec.DistributionGenerator import DistributionGenerator

class Netwld2v:
    r"""An implementation of whiolegraph emebdding based on doc2vec and WL algorithm
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        annotation (str): type of node (annotation) label used to build documents (default is node degree)
    """
    def __init__(self, wl_iterations: int=2, vertex_attribute=None, dimensions: int=128, annotation: str="degree",
                 workers: int=4, down_sampling: float=0.0001, epochs: int=10, 
                 learning_rate: float=0.025, min_count: int=5, seed: int=42, verbose=False):

        self.wl_iterations = wl_iterations
        assert self.wl_iterations > 0, "WL recursions must be > 0"
        self.verbose = verbose
        self.dimensions = dimensions
        assert self.dimensions > 0, "WL recursions must be > 0"
        self.vertex_attribute = vertex_attribute
        self.annotation = annotation
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def __check_graphs(self, graphs):
        """Checking the consecutive numeric indexing."""
        for graph in graphs:
           numeric_indices = [index for index in range(graph.vcount())]
           node_indices = sorted([node.index for node in ig.VertexSeq(graph)])
           assert numeric_indices == node_indices, "The node indexing is wrong."

    def __set_features(self, graphs, annotation):
        """ compute probabilities and initilize feature of graph nodes"""
        self.probmats = DistributionGenerator(self.annotation[0], graphs, verbose=self.verbose).get_distributions()
        for gidx,graph in enumerate(graphs):
            for v in ig.VertexSeq(graph):
                v["feature"]= np.argmax(self.probmats[gidx][v.index])   
        
    def fit(self, graphs: List[ig.Graph]):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of igraph graphs)* - The graphs to be embedded.
        """
        self.__check_graphs(graphs)    # check graphs conditions
        self.__set_features(graphs, self.annotation)
        documents = [WeisfeilerLehman(graph, self.wl_iterations, self.vertex_attribute, self.annotation) for graph in graphs]
        documents = [TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(documents)]

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        epochs=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        self._embedding = [model.docvecs[str(i)] for i, _ in enumerate(documents)]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
