#!/usr/bin/python

import networkx as nx
import numpy as np
import pdb
import sys
import os
import json
import numbers
import importlib
import itertools
from math import log10


import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
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
		self._hyp_d = {}


	def search_space(self, hyp_range):
		### actual number
		#space = {k: (min(v), max(v)) for k, v in hyp_range.items() if isinstance(v[0], numbers.Number)}
		### power of 10
		space = {k: (log10(min(v)), log10(max(v))) for k, v in hyp_range.items() if isinstance(v[0], numbers.Number)}
		category_para = {k: v for k, v in hyp_range.items() if not isinstance(v[0], numbers.Number)}
		return space, category_para


	def optimization_func(self, **hyp_space):
		## method class	
		MethClass = getattr(
			importlib.import_module("gem.embedding.%s" % self._meth),
			methClassMap[self._meth])
		self._hyp_d.update({"d": self._dim})

		## actual number
		#self._hyp_d.update(hyp_space)
		## turn power into actual number\
		hyp_space = {k:10**v for k, v in hyp_space.items()}
		print("current hyp_space value is", hyp_space)

		self._hyp_d.update(hyp_space)
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
			self._hyp_d.update({"data_set": self._data_set})
		print("hyp_d:",self._hyp_d)
		MethObj = MethClass(self._hyp_d)
		gr, lp, nc = exp.run_exps(MethObj, self._meth, self._dim, self._di_graph,
				self._data_set, self._node_labels, self._params)
		res = np.mean(lp)
		print("lp res", res)
		return res



	def optimize(self, random_state = 5, verbose = 2, init_points = 10, n_iter = 5, acq = 'poi' ):
        ## ei, poi, ucb
		#utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

		for hyp in itertools.product(*self._category_para.values()):
			self._hyd_d = dict(zip(self._category_para.keys(),hyp))
			print("category_para: ",self._hyd_d )
			optimizer = BayesianOptimization(
			f = self.optimization_func,
			pbounds = self._search_space,
			random_state = random_state
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
				n_iter=n_iter,
				acq= acq,
				kappa=2.576,
				xi=1.0
			)


			print("category_para: ", )
			print("Final result:", optimizer.max)
		return 0