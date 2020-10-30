import hashlib
import networkx as nx
import igraph as ig
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import netpro2vec.utils as utils

class WeisfeilerLehman(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
    """
    def __init__(self, graph: ig.Graph, wl_iterations: int, vertex_attribute: None, annotation: str, verbose: bool=False):
        """
        Initialization method which also executes feature extraction.
        """
        self.verbose = verbose
        self.tqdm = tqdm if self.verbose else utils.nop
        self.wl_iterations = wl_iterations
        assert self.wl_iterations > 0, "WL recursions must be > 0"
        self.annotation = annotation
        self.graph = graph
        self.mode = "OUT" if graph.is_directed() else 'ALL' 
        self.vertex_attribute = vertex_attribute
        self.vertex_attribute_list = self.__get_vertex_attributes()
        self._set_features()
        self._do_recursions()

    def __get_vertex_attributes(self):
        if self.vertex_attribute is None:     # if no attribute is specifyed, use the index as vertex label 
           return [v.index for v in ig.VertexSeq(self.graph)]
        elif self.vertex_attribute in self.graph.vs.attributes():
           return [v[self.vertex_attribute] for v in ig.VertexSeq(self.graph)]
        else:
           raise Exception('The graph does not have the provided vertex ')

    def _set_features(self):
        """
        Creating the features.
        """
        self.extracted_features = [ (self.vertex_attribute_list[v.index], v["feature"]) for v in ig.VertexSeq(self.graph) ]
        #print("INIT", self.extracted_features)

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in ig.VertexSeq(self.graph):
            nebs = self.graph.neighbors(node, mode=self.mode)
            degs = [neb["feature"] for neb in self.graph.vs[nebs]]
            print(nebs)
            print(degs)
            features = [str(node["feature"])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            node["feature"] = hashing
            #node["feature"] = features
            new_features[node.index] = hashing
        #self.extracted_features = {k: self.extracted_features[k] + [v] for k,v in new_features.items()}
        self.extracted_features = [ (self.vertex_attribute_list[k], v) for k,v in new_features.items() ]
        #print(self.extracted_features)
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [feature for node,features in self.extracted_features for feature in features]



