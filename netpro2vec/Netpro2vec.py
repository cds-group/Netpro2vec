# @Time    : 08/09/2020 09:09
# @Author  : Ichcha Manipur
# @Email   : ichcha.manipur@gmail.com
# @File    : Netprob2vecmulti.py

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from joblib import wrap_non_picklable_objects
from joblib import parallel_backend
from netpro2vec.ProbDocExtractor import ProbDocExtractor
from netpro2vec.DistributionGenerator import DistributionGenerator, \
	probability_aggregator_cutoff
from netpro2vec import utils
from gensim.utils import simple_preprocess
from gensim import models, corpora
import numpy as np
import itertools
import igraph as ig
from typing import *
import os
import pickle as pk

"""
Netrpo2vec.py
====================================
The core module of Netpro2vec software.
"""

class Netpro2vec:
	"""The class implementation of Netpro2vec model for whole-graph embedding.
    from the IEEE TCBB '20 paper "Netpro2vec: a Graph Embedding Framework for Biological Networks". The procedure
    uses probability distribution representations of graphs and skip-gram learning modelor later.

    Args:
        **format** *(str, optional)* -  graph format. Dfault is "graphml"

        **dimensions** *(int, optional)* – number of features. Default is 128.

        **prob_type** *(list of str, optional)* –  list of probability types. Default is ["tm1"] (allowed values: "ndd", "tm<walk-distance>").
        "ndd" means that a word for each node is formed by concatenating labels (or node ids) of nodes whose distance above a cut-off threshold with respect to the source node.
        "tm<walf-distance>" means that a word for each node is formed  by concatenating node labels (or node ids) which are reacheable by the source node
        with a transition probability above a cut-off threshold. 

        **cut_off** *(list of float, optional)* –  list of cut-off thresholds to form words for each node. Default is [0.01].
        Only the labels (or ids) of nodes with transition probability (for "tm<walk-distance>") or with distance (for "ndd") 
        above this threshold are used for building the word.

        **agg_by** *(list of int, optional)* – list of *aggregators* in words formed with "ndd" annotation. Default is [0].
        The number of bins used to build the Node Distance Distribution histogram is scaled down by
        the "agg_by" value. A unity value means that the number of bins is set to the maximum diameter.
        A value *n* means that the number of bins is equal to *max-diameter/n*.
        A zero value means that this paramter does not apply to the annotation (i.e. "tm<walk-distance>").   

        **extractor** *(list of int, optional)* – list of *extractor modes* in builidng words, one modality for each specified annotation. 
        Default is [0]. Currently, only extractor 1 and 6 are supported. 
        With respect to the corresponding annotation in the **prob_type** list: 
        - extractor 1 returns words only from a single cut off. 
        - extractor 2 returns multiple words of different lengths from different probability cut offs (The set of predefinied cut offs is: 0, 0.1, 0.3, 0.5).

        **min_count** *(int, optional)* – Ignores all words with total frequency lower than this (Doc2Vec). Default is 5

        **workers** *(int, optional)* – use these many worker threads to train the model (Doc2Vec). Default is 4

        **epochs** *(int, optional)* – Number of iterations (epochs) over the corpus (Doc2Vec). Default is 10

        **remove_inf** *(bool, optional)*: flag for removal of infinity value in histogram bins. Default is False.

        **vertex_labels** *(bool, optional)*: flag to set if graphs have vertex labels to be considered. Default is False

        **encodew** *(bool, optional)*: flag to unset if graph words are encoded into hash keys. Default is True

        **save_probs *(bool, optional)*: flag to enable probability distribution saving. Default is False

        **load_probs *(bool, optional)*: flag to enable probability distribution restoring. Default is False

        **save_vocab *(bool, optional)*: flag to enable vocabulary saving. Default is False

        **load_vocab *(bool, optional)*: flag to enable vocabulary restoring. Default is False
    """

	def __init__(self, format="graphml", dimensions=128, prob_type: List[str]=["tm1"], 
		         extractor=[1],cut_off=[0.01], agg_by=[0],
		         min_count=5, down_sampling=0.0001,workers=4, epochs=10, learning_rate=0.025, 
				 remove_inf=False, vertex_attribute=None, seed=0,
				 verbose=False,encodew=True,
				 save_probs=False, load_probs=False, save_vocab=False, load_vocab=False):
		"""Creatinng the model."""
		if len({len(i) for i in [prob_type,extractor,cut_off,agg_by]}) != 1:
			raise Exception("Probability type list must be equal-sized wrt aggregator and cutoff arguments")
		for i,a in enumerate(agg_by):
			if prob_type[i] == "ndd" and (int(a) < 1 or int(a) > 10):	
				raise Exception("Aggregators values for %s must be in the range [1,10]"%prob_type[i])
			if prob_type[i] != "ndd" and int(a) != 0:
				print(a)
				raise Exception("Aggregators values for %s must be 0 (disabled)"%prob_type[i])
		if any(e != 1 and e != 2 for (e) in extractor):	
			raise Exception("Supported extractor modes are 1 (single cut-off) and 2 (multiple cut-offs)")
		if dimensions < 0: 
			raise Exception("Dimensions must be >0 (default 128)")
		if format not in ["graphml", "edgelist"]: 
			raise Exception("graph format can be graphml or edgelist (default graphml)")
		self.prob_type = prob_type
		self.extractor = extractor
		self.cut_off = cut_off
		self.agg_by = agg_by
		self.dimensions = dimensions
		self.remove_inf = remove_inf
		self.vertex_attribute = vertex_attribute
		self.vertex_attribute_list = []
		self.embedding = None
		self.probmats = {}
		self.encodew=encodew
		self.saveprobs = save_probs
		self.loadprobs = load_probs
		self.savevocab = save_vocab
		self.loadvocab = load_vocab
		if not os.path.exists('.np2vec'):
			os.makedirs('.np2vec')
		self.probmatfile = os.path.join('.np2vec','probmats.pkl')
		self.vocabfile = os.path.join('.np2vec','vocab.pkl')
		self.min_count=min_count 
		self.down_sampling=down_sampling
		self.workers=workers 
		self.epochs=epochs
		self.learning_rate=learning_rate
		self.randomseed=seed
		self.document_collections = []
		self.document_collections_list = []
		self.verbose = verbose
		self.tqdm = tqdm if self.verbose else utils.nop
		self.model = None

	def get_dictionary_corpus(self):
		# for tfidf and lda
		diction = corpora.Dictionary(
			[simple_preprocess(" ".join(line)) for line in
			 self.document_collections_list[-1]])
		corpus = [diction.doc2bow(simple_preprocess(" ".join(line))) for line in
				  self.document_collections_list[-1]]
		return diction, corpus


	def get_sentences(self, graphs: List[ig.Graph]):
		probmats = self.__generate_probabilities_newsample(graphs)
		if self.vertex_attribute is not None:
			self.get_vertex_attributes(graphs)
		docs = self.__get_document_collections_newsample(probmats, encodew=self.encodew)
		return docs
		
	def get_documents(self, graphs: List[ig.Graph]):
		"""Document generator method of Netpro2vec model.

	    Args:
	        **graphs** *(List igraph.Graph objs)* - list of graphs in igraph format types.

	    Return:
			The lisy of documents representing the graphs.
	    """
		self.num_graphs = len(graphs)
		self.__generate_probabilities(graphs)
		if self.vertex_attribute is not None:
			self.get_vertex_attributes(graphs)
		self.__get_document_collections(tag_doc=False,encodew=self.encodew)
		return [ " ".join(doc) for doc in self.document_collections_list[-1]]

	def fit(self, graphs: List[ig.Graph]):
		"""Fitting method of Netpro2vec model.

	    Args:
	        **graphs** *(List igraph.Graph objs)* - list of graphs in igraph format types.

	    Return:
			The trained **Netpro2vec** model.
	    """
		self.format = format
		self.num_graphs = len(graphs)
		if self.loadvocab:
			try:
				utils.vprint("Loading vocabulary...", end='\n', verbose=self.verbose)
				infile = open(self.vocabfile,'rb')
				self.document_collections_list = pk.load(infile)
				infile.close()
			except Exception as e:
				utils.vprint("Cannot load vocabulary...%s"%e, end='\n', verbose=self.verbose)
				utils.vprint("...Let's generate it by scratch!", end='\n', verbose=self.verbose)
				self.__generate_probabilities(graphs)
				if self.vertex_attribute is not None:
					self.get_vertex_attributes(graphs)
				self.__get_document_collections(encodew=self.encodew)
		else:
			self.__generate_probabilities(graphs)
			if self.vertex_attribute is not None:
				self.get_vertex_attributes(graphs)
			self.__get_document_collections(encodew=self.encodew)
		if self.savevocab:
			try:
				utils.vprint("Saving vocabulary...", end='\n', verbose=self.verbose)
				outfile = open(self.vocabfile,'wb')
				pk.dump(self.document_collections_list, outfile)
				outfile.close()
			except Exception as e:
				raise Exception("Cannot save vocabulary...",e, e.args)
		self.__run_d2v(dimensions=self.dimensions,min_count=self.min_count,down_sampling=self.down_sampling,
					workers=self.workers, epochs=self.epochs, learning_rate=self.learning_rate)
		return self

	def infer_vector(self, graphs: List[ig.Graph], steps=None, alpha=None):
		"""Inferring embedding method of Netpro2vec model.

	    Args:
	        **graphs** *(List igraph.Graph objs)* - list of graphs in igraph format types.

	    Return:
			The trained **Netpro2vec** model.
	    """
		probmats = self.__generate_probabilities_newsample(graphs)
		if self.vertex_attribute is not None:
			self.get_vertex_attributes(graphs)
		docs = self.__get_document_collections_newsample(probmats, encodew=self.encodew)
		idx = 0
		if len(self.prob_type) > 1:
			idx = len(self.prob_type) - 1
		documents = [ d.words for d in docs[idx]]
		if steps is None: steps = self.epochs 
		if alpha is None: alpha = self.learning_rate
		utils.vprint("Doc2Vec inferring (steps=%d, alpha%f) in progress..."%(steps,alpha), end='', verbose=self.verbose)
		embedding_list = [self.model.infer_vector(doc,steps=steps,alpha=alpha) for doc in documents]
		utils.vprint("Done!", verbose=self.verbose)
		return embedding_list

	def get_embedding(self):
		"""Access embedding of Netpro2vec model.

	    Return:
			The produced embedding in numpy array format (None if the mode was not trained).
	    """
		return np.array(self.embedding)

	def get_memberships(self):
		"""Access last document list used for training the Netpro2vec model.

	    Return:
			The document list used in last model training ([] if the model was never trained).
	    """
		return np.array(self.document_collections_list[-1])

	def __generate_probabilities(self, graphs: List[ig.Graph]):
		if self.loadprobs:
			try:
				utils.vprint("Loading prob mats...", end='\n', verbose=self.verbose)
				infile = open(self.probmatfile,'rb')
				self.probmats = pk.load(infile)
				infile.close()
			except Exception as e:
				utils.vprint("Cannot load saved probs...%s"%e, end='\n', verbose=self.verbose)
				utils.vprint("...Let's generate them by scratch!", end='\n', verbose=self.verbose)
				for prob_type in self.prob_type:
					self.probmats[prob_type] = DistributionGenerator(prob_type, graphs,verbose=self.verbose).get_distributions()
		else:
			for prob_type in self.prob_type:
				self.probmats[prob_type] = DistributionGenerator(prob_type, graphs,verbose=self.verbose).get_distributions()
			if self.saveprobs:
				try:
					utils.vprint("Saving prob mats...", end='\n', verbose=self.verbose)
					outfile = open(self.probmatfile,'wb')
					pk.dump(self.probmats, outfile)
					outfile.close()
				except Exception as e:
					raise Exception("Cannot save probs...",e, e.args)

	def __generate_probabilities_newsample(self, graphs: List[ig.Graph]):
		probmats = {}
		for prob_type in self.prob_type:
			probmats[prob_type] = DistributionGenerator(prob_type, graphs,verbose=self.verbose).get_distributions()
		return probmats

	@delayed
	@wrap_non_picklable_objects
	def __batch_feature_extractor(self, probability_distrib_matrix, name, word_tag=None, tag=True,
								aggregate=0, cut=0, encodew=True, extractor=1):
		''' Generate a document collection describing a single graph
		    by concatenating distribution matrix data 
		    (NDD or TM<int>) for each node in the graph. 
		'''
		if self.vertex_attribute is not None:
			vertex_labels = self.vertex_attribute_list[int(name)]
		else:
			vertex_labels = None
		if aggregate > 0 or cut > 0:
			probability_distrib_matrix = probability_aggregator_cutoff(
				probability_distrib_matrix, cut_off=cut,
				agg_by=aggregate, return_prob=True,
				remove_inf=self.remove_inf)

		document_collections = ProbDocExtractor(probability_distrib_matrix,
													name, word_tag,
													extractor=extractor,
													tag=tag, encodew=encodew,
													vertex_labels=vertex_labels)
		# perfrom Garbage Collecting
		#del probability_distrib_matrix
		#gc.collect()         
		return document_collections

	def get_vertex_attributes(self, graphs):
		if self.vertex_attribute in graphs[0].vs.attributes():
			self.vertex_attribute_list = [graphs[x].vs[self.vertex_attribute] for x in range(0, len(graphs))]
		else:
			raise Exception('The graph does not have the provided vertex '
							'attribute in -A! ')

	def __get_document_collections(self, workers=4, tag_doc=True, encodew=True):
		''' Generate documents for graphs. 
		    If multiple distributions are specified, documents are generated for each
		    distribution, then merged ito una single vocabulary.
		'''
		document_collections_all = []
		for prob_idx,prob_type in enumerate(self.prob_type):
			if (len(self.prob_type) > 1) or (tag_doc is False):
				tag = False
			else:
				tag = True
			prob_mats = self.probmats[prob_type]
			utils.vprint("Building vocabulary for %s..."%prob_type, verbose=self.verbose)			
			with parallel_backend('threading', n_jobs=workers):
				document_collections = Parallel()(self.__batch_feature_extractor(p, str(i), prob_type, tag=tag,
											extractor=self.extractor[prob_idx],
											encodew=encodew,
											cut=self.cut_off[prob_idx],
											aggregate=self.agg_by[prob_idx])
												  for i, p in enumerate(self.tqdm(prob_mats)))
			document_collections = [document_collections[
										x].graph_document for x in
									range(0, len(document_collections))]
			document_collections_all.append(document_collections)
		# merge multiple documents into one vocabulary
		if len(self.prob_type) > 1:
			doc_merge = [e for e in zip(*document_collections_all)]
			merged_document = [list(itertools.chain.from_iterable(doc)) for doc in doc_merge]
			document_collections_all.append(doc_merge)
			if tag_doc:
				for prob_type_doc in document_collections_all:
					self.document_collections_list.append([
						models.doc2vec.TaggedDocument(
							words=prob_type_doc[
							x], tags=["g_%d"%x]) for x in range(
							0, len(prob_type_doc))])
			else:
				self.document_collections_list = document_collections_all
		else:
			self.document_collections_list.append(document_collections)

	def __get_document_collections_newsample(self, probmats, workers=4, tag_doc=True, encodew=True):
		''' Generate documents for graphs. 
		    If multiple distributions are specified, documents are generated for each
		    distribution, then merged ito una single vocabulary.
		'''
		document_collections_list = []
		document_collections_all = []
		for prob_idx,prob_type in enumerate(self.prob_type):
			if (len(self.prob_type) > 1) or (tag_doc is False):
				tag = False
			else:
				tag = True
			prob_mats = probmats[prob_type]
			utils.vprint("Building vocabulary for %s..."%prob_type, verbose=self.verbose)			
			with parallel_backend('threading', n_jobs=workers):
				document_collections = Parallel()(self.__batch_feature_extractor(p, str(i), prob_type, tag=tag,
											extractor=self.extractor[prob_idx],
											encodew=encodew,
											cut=self.cut_off[prob_idx],
											aggregate=self.agg_by[prob_idx])
												  for i, p in enumerate(self.tqdm(prob_mats)))
			document_collections = [document_collections[
										x].graph_document for x in
									range(0, len(document_collections))]
			document_collections_all.append(document_collections)
		# merge multiple documents into one vocabulary
		if len(self.prob_type) > 1:
			doc_merge = [e for e in zip(*document_collections_all)]
			merged_document = [list(itertools.chain.from_iterable(doc)) for doc in doc_merge]
			document_collections_all.append(doc_merge)
			if tag_doc:
				for prob_type_doc in document_collections_all:
					document_collections_list.append([
						models.doc2vec.TaggedDocument(
							words=prob_type_doc[
							x], tags=["g_%d"%x]) for x in range(
							0, len(prob_type_doc))])
			else:
				document_collections_list = document_collections_all
		else:
			document_collections_list.append(document_collections)
		return document_collections_list

	def __get_diction_corpus(self):
		# for tfidf and lda
		diction = corpora.Dictionary(
			[simple_preprocess(" ".join(line)) for line in
			 self.document_collections])
		corpus = [diction.doc2bow(simple_preprocess(" ".join(line))) for line in
				  self.document_collections]
		return diction, corpus

	def __run_d2v(self, dimensions=128, min_count=5,down_sampling=0.0001,
					workers=4, epochs=10, learning_rate=0.025):
		''' Run Doc2Vec library to:
		    1) produce the vocabulary from the list of documents representing the graphs
		    2) train the model on the vocabulary.
		    3) produce the embedding matrix, then store in the "embedding" attribute of the 
		    Netpro2vec object.
		'''
		idx = 0
		if len(self.prob_type) > 1:
			idx = len(self.prob_type) - 1
		utils.vprint("Doc2Vec embedding in progress...", end='', verbose=self.verbose)
		model = models.doc2vec.Doc2Vec(self.document_collections_list[idx],
						vector_size=dimensions,
						window=0,
						min_count=min_count,
						dm=0,
						sample=down_sampling,
						workers=workers,
						epochs=epochs,
						alpha=learning_rate,
						seed=self.randomseed)
		self.model = model
		utils.vprint("Done!", verbose=self.verbose)
		self.embedding = model.docvecs.doctag_syn0

