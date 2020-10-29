import hashlib
import networkx as nx
import igraph as ig
import numpy as np
from typing import List, Dict

class WeisfeilerLehman:
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
    """
    def __init__(self, graph: ig.Graph, wl_iterations: int, vertex_attribute: None, annotation: str):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.annotation = annotation
        self.graph = graph
        self.vertex_attribute = vertex_attribute
        self.vertex_attribute_list = []
        self._set_features()
        self._do_recursions()

    def __get_vertex_attributes(self, graphs):
        if self.vertex_attribute is None:     # if no attribute is specifyed, use the index as vertex label 
           self.vertex_attribute_list = [v.index for v in ig.VertexSeq(self.graph)]
        elif self.vertex_attribute in graphs[0].vs.attributes():
           self.vertex_attribute_list = [v[self.vertex_attribute] for v in ig.VertexSeq(self.graph)]
        else:
           raise Exception('The graph does not have the provided vertex ')

    def _set_features(self):
        """
        Creating the features.
        """
        self.extracted_features = [ (self.vertex_attribute_list[v.index], v["feature"]) for v in ig.VertexSeq(self.graph) ]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [neb["feature"] for neb in nebs]
            features = [str(node["feature"])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            node["feature"] = hashing
            new_features[node] = hashing
        self.extracted_features = {k: self.extracted_features[k] + [v] for k, v in new_features.items()}
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        assert self.wl_iterations > 0, "WL recursions must be > 0"
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
        return [feature for node, features in self.extracted_features.items() for feature in features]

