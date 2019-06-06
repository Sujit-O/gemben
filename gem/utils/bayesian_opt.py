#!/usr/bin/python

import networkx as nx
import numpy as np
import pdb
import sys
import os
import json
import numbers
import importlib


import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
###### how to import run_exps??
from gem.experiments import exp

methClassMap = {"gf": "GraphFactorization",
				"hope": "HOPE",
				"lap": "LaplacianEigenmaps",
				"node2vec": "node2vec",
				"sdne": "SDNE",
				"pa": "PreferentialAttachment",
				"rand": "RandomEmb",
				"cn": "CommonNeighbors",
				"aa": "AdamicAdar",
				"jc": "JaccardCoefficient"}
expMap = {"gf": "GF MAP", "lp": "LP MAP",
			"nc": "NC MAP"}


class BayesianOpt(object):
	""" 
	bayesian global optimization with Gaussian Process
	"""
	def __init__(self, *args, **kwargs):
		for key in kwargs.keys():
			self.__setattr__('_%s' % key,kwargs[key])
		self._params = kwargs
		## when method is a list??
		self._meth = self._methods[0]
		## when dim is a list??
		self._dim = int(self._dimensions[0])
		self._search_space, self._category_para = self.search_space(self._model_hyp_range[self._meth])


	def search_space(self, hyp_range):
		space = {k: (min(v), max(v)) for k, v in hyp_range.items() if isinstance(v[0], numbers.Number)}
		category_para = {k: v for k, v in hyp_range.items() if not isinstance(v[0], numbers.Number)}
		return space, category_para



	def optimization_func(self, **hyp_space):
		## method class	
		MethClass = getattr(
			importlib.import_module("gem.embedding.%s" % self._meth),
			methClassMap[self._meth])
		hyp_d = {"d": self._dim}
		hyp_d.update(hyp_space)
		if self._meth == "sdne":
			hyp_d.update({
				"modelfile": [
					"gem/intermediate/enc_mdl_%s_%d.json" % (self._data_set, self._dim),
					"gem/intermediate/dec_mdl_%s_%d.json" % (self._data_set, self._dim)
				],
				"weightfile": [
					"gem/intermediate/enc_wts_%s_%d.hdf5" % (self._data_set, self._dim),
					"gem/intermediate/dec_wts_%s_%d.hdf5" % (self._data_set, self._dim)
				]
			})
		elif self._meth == "gf" or self._meth == "node2vec":
			hyp_d.update({"data_set": self._data_set})
		print("hyp_d:",hyp_d)
		MethObj = MethClass(hyp_d)
		gr, lp, nc = exp.run_exps(MethObj, self._meth, self._dim, self._di_graph,
				self._data_set, self._node_labels, self._params)
		return np.mean(lp)



	def optimize(self, random_state = 1, verbose = 2, init_points = 2, n_iter = 5 ):

		if not self._category_para:
			optimizer = BayesianOptimization(
				f = self.optimization_func,
				pbounds = self._search_space,
				random_state = random_state,
			)

		log_path = "gem/intermediate/bays_opt/"
		try:
			os.makedirs(log_path)
		except:
			pass
		logger = JSONLogger(path=log_path+"logs.json")
		optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

		optimizer.maximize(
			init_points=init_points, 
			n_iter=n_iter
		)

		print("Final result:", optimizer.max)
		return 0