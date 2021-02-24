# @Time    : 13/08/2020 19:11
# @Author  : Ichcha Manipur
# @Email   : ichcha.manipur@gmail.com
# @File    : ProbDocExtractor.py


import numpy as np
import hashlib
from gensim.models.doc2vec import TaggedDocument
import itertools
import pandas as pd


class ProbDocExtractor:
	"""
	Feature extractor class for Netprob2vec
	Convert graph probability matrix to a document
	specify extractor for ordered/unordered/rounded/tagged/untagged
	"""
	def __init__(self, probability_distrib_matrix, doc_tag, word_tag=None,
				 extractor=1, tag=True, encodew=True, vertex_labels=None):
		self.probability_distrib_matrix = probability_distrib_matrix
		self.word_tag = word_tag
		self.doc_tag = doc_tag
		self.encode = encodew
		self.features_graph = []
		self.hashing_list = []
		self.graph_document = []
		self.extractor = extractor
		self.tag = tag
		self.vertex_labels = vertex_labels
		if self.extractor == 1:
			self.ordered_probability_extractor()
		elif self.extractor == 2:
			self.ordered_probability_extractor_multi()
		else:
			raise Exception("extractor level is in the range [1, 2]")

	def ordered_probability_extractor(self):
		"""
		Ordered sequence (decreasing probability) of nodes or bin index (of
		distances) for each rooted node converted to words
		This is for networks edge-weighted with real data like gene(perhaps not
		for weights with stat data like correlation or p-values - To be tested).
		"""

		probability_distrib_matrix = np.fliplr(np.argsort(
			self.probability_distrib_matrix, axis=1))
		features_graph = probability_distrib_matrix.tolist()
		cut = (self.probability_distrib_matrix > 0).sum(axis=1)
		self.features_graph = [features_graph[i][0:cut[i]] for i in range(0,
																	  len(cut))]
		self.get_graph_document()

	def ordered_probability_extractor_multi(self):
		"""
		Ordered sequence (decreasing probability) of nodes or bin index (of
		distances) for each rooted node converted to words
		This is for networks edge-weighted with real data like gene(perhaps not
		for weights with stat data like correlation or p-values - To be tested).
		"""
		cutoff_list = [0, 0.1, 0.3, 0.5]
		tag = self.tag
		self.tag = False
		graph_document = []
		unique_words = []
		for cut_off in cutoff_list:
			self.probability_distrib_matrix[self.probability_distrib_matrix <
									   cut_off] = 0

			probability_distrib_matrix = np.fliplr(np.argsort(
				self.probability_distrib_matrix, axis=1))
			features_graph = probability_distrib_matrix.tolist()
			cut = (self.probability_distrib_matrix > 0).sum(axis=1)
			self.features_graph = [features_graph[i][0:cut[i]] for i in range(0,
																	  len(cut))]
			self.get_graph_document()
			graph_document.append(self.graph_document)
		node_features_merge = [e for e in zip(*graph_document)]
		merged_document = list(itertools.chain.from_iterable(node_features_merge))
		[unique_words.append(item) for item in merged_document if item not in
		 unique_words]
		merged_document = unique_words
		if tag:
			merged_document = self.get_tagged_doc(merged_document)
		self.graph_document = merged_document
		self.probability_distrib_matrix = []
		self.features_graph = []

	def get_graph_document(self):
		if self.word_tag is not None:
			if self.vertex_labels is not None and 'tm' in self.word_tag:
				vertex_labels = self.vertex_labels
				self.features_graph = [[vertex_labels[i] for i in
									self.features_graph[x]]
								  for x in range(0, len(self.features_graph))]
				[self.features_graph[i].insert(0, self.word_tag + '_' +
											   str(vertex_labels[i])) for i in
				 range(0, len(vertex_labels))]
			elif self.vertex_labels is not None and self.word_tag == 'ndd':
				vertex_labels = self.vertex_labels
				[self.features_graph[i].insert(0, self.word_tag + '_' + str(
					vertex_labels[i]))
				 for i in range(0, len(vertex_labels))]
			elif self.vertex_labels is None:
				[self.features_graph[i].insert(0, self.word_tag + '_' + str(
					i)) for i in range(0, len(self.features_graph))]
		new_ind_list_node_seq = [("_".join(str(item) for item in new_ind)) for
								 new_ind in self.features_graph]

		# # uncomment to keep only unique words
		# unique_words = []
		# [unique_words.append(item) for item in new_ind_list_node_seq if item not in
		#  unique_words]
		# new_ind_list_node_seq = unique_words
		# print(new_ind_list_node_seq)

		if self.encode:
			hash_object_list = [hashlib.md5(features.encode()) for features in
								new_ind_list_node_seq]
			self.graph_document = [hash_object.hexdigest() for hash_object in
							hash_object_list]
		else:
			self.graph_document = new_ind_list_node_seq

		if self.tag:
			self.graph_document = self.get_tagged_doc(self.graph_document)
		if self.extractor <= 5:
			self.probability_distrib_matrix = []
			self.features_graph = []

	def get_graph_document_split(self, encode=True):
		new_ind_list_node_seq = []
		for i in range(0, len(self.features_graph)):
			new_ind_list_node_seq.extend([self.word_tag + str(i) + '_' + str(
				new_ind) for new_ind in self.features_graph[i]])

		if self.encode:
			hash_object_list = [hashlib.md5(features.encode()) for features in
								new_ind_list_node_seq]
			self.graph_document = [hash_object.hexdigest() for hash_object in
							hash_object_list]
		else:
			self.graph_document = new_ind_list_node_seq

		if self.tag:
			self.graph_document = self.get_tagged_doc(self.graph_document)

		self.probability_distrib_matrix = []
		self.features_graph = []

	def get_tagged_doc(self, hashing_list):
		tagged_graph_document = TaggedDocument(words=hashing_list,
										   tags=["g_" + self.doc_tag])
		return tagged_graph_document
