import logging
from copy import deepcopy
import traceback


import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm
import scipy.linalg as spla
import scipy.stats as sps
import scipy.optimize as spo
from sklearn.ensemble import RandomForestRegressor

from hpbandster.core.base_config_generator import base_config_generator


class RandomForrestWithVariance(RandomForestRegressor):

	def predict(self, X):
		X = np.atleast_2d(X)

		all_predictions = [tree.predict(X) for tree in self.estimators_]
		y_hat = sum(all_predictions) / self.n_estimators
		y_var = np.var(all_predictions, axis=0, ddof=1)
		return y_hat, y_var

class RandomForestEI(base_config_generator):
	def __init__(self, configspace, min_points_in_model = None,
				 top_n_percent=15, num_samples = 64, random_fraction=1/3,
				 bandwidth_factor=3, min_bandwidth=1e-3,
				**kwargs):
		"""
			Fits for each given budget a kernel density estimator on the best N percent of the
			evaluated configurations on this budget.


			Parameters:
			-----------
			configspace: ConfigSpace
				Configuration space object
			top_n_percent: int
				Determines the percentile of configurations that will be used as training data
				for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
				for training.
			min_points_in_model: int
				minimum number of datapoints needed to fit a model
			num_samples: int
				number of samples drawn to optimize EI via sampling
			random_fraction: float
				fraction of random configurations returned
			bandwidth_factor: float
				widens the bandwidth for contiuous parameters for proposed points to optimize EI
			min_bandwidth: float
				to keep diversity, even when all (good) samples have the same value for one of the parameters,
				a minimum bandwidth (Default: 1e-3) is used instead of zero. 

		"""
		super().__init__(**kwargs)
		self.configspace = configspace

		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:
			self.min_points_in_model = len(self.configspace.get_hyperparameters())+1
		
		if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
			self.logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.configspace.get_hyperparameters())+1))
			self.min_points_in_model =len(self.configspace.get_hyperparameters())+1
		
		self.num_samples = num_samples
		self.random_fraction = random_fraction
		self.top_n_percent = top_n_percent

		hps = self.configspace.get_hyperparameters()

		self.kde_vartypes = ""
		self.vartypes = []

		for h in hps:
			if hasattr(h, 'sequence'):
				raise RuntimeError('This version on GPEIMCMCBO does not support ordinal hyperparameters. Please encode %s as an integer parameter!'%(h.name))
			
			if hasattr(h, 'choices'):
				self.kde_vartypes += 'u'
				self.vartypes +=[ len(h.choices)]
			else:
				self.kde_vartypes += 'c'
				self.vartypes +=[0]
		
		self.vartypes = np.array(self.vartypes, dtype=int)

		# store precomputed probs for the categorical parameters
		self.cat_probs = []
		
		self.noiseless = False
		self.randomforest = RandomForrestWithVariance()

		self.configs = dict()
		self.losses = dict()
		self.good_config_rankings = dict()


	def largest_budget_with_model(self):
		if len(self.kde_models) == 0:
			return(-float('inf'))
		return(max(self.kde_models.keys()))

	def get_config(self, budget):
		"""
			Function to sample a new configuration

			This function is called inside Hyperband to query a new configuration


			Parameters:
			-----------
			budget: float
				the budget for which this configuration is scheduled

			returns: config
				should return a valid configuration

		"""
		
		self.logger.debug('start sampling a new configuration.')
		

		sample = None
		info_dict = {}
		
		best = np.inf
		if budget in self.losses:
			best = np.min(self.losses[budget])

		if len(self.losses) < 1:
			sample = self.configspace.sample_configuration().get_dictionary()
			info_dict['model_based_pick'] = True
			return sample, info_dict


		current_configs = []
		y_hats = []
		y_vars = []
		for i in range(1000):
			current_config = self.configspace.sample_configuration()
			y_hat, y_var = self.randomforest.predict([current_config.get_array()])
			current_configs.append(current_config.get_dictionary())
			y_hats.append(y_hat)
			y_vars.append(y_var)
		
		
		# Expected improvement
		func_s = np.sqrt(y_vars) + 0.0001
		u = (best - y_hats) / func_s
		ncdf = sps.norm.cdf(u)
		npdf = sps.norm.pdf(u)
		ei = func_s*( u*ncdf + npdf)

		best_sample = np.argmax(ei)
		sample = current_configs[best_sample]
		info_dict['model_based_pick'] = True
		return sample, info_dict


	def impute_conditional_data(self, array):

		return_array = np.empty_like(array)

		for i in range(array.shape[0]):
			datum = np.copy(array[i])
			nan_indices = np.argwhere(np.isnan(datum)).flatten()

			while (np.any(nan_indices)):
				nan_idx = nan_indices[0]
				valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()

				if len(valid_indices) > 0:
					# pick one of them at random and overwrite all NaN values
					row_idx = np.random.choice(valid_indices)
					datum[nan_indices] = array[row_idx, nan_indices]

				else:
					# no good point in the data has this value activated, so fill it with a valid but random value
					t = self.vartypes[nan_idx]
					if t == 0:
						datum[nan_idx] = np.random.rand()
					else:
						datum[nan_idx] = np.random.randint(t)

				nan_indices = np.argwhere(np.isnan(datum)).flatten()
			return_array[i,:] = datum
		return(return_array)

	def new_result(self, job, update_model=True):
		"""
			function to register finished runs

			Every time a run has finished, this function should be called
			to register it with the result logger. If overwritten, make
			sure to call this method from the base class to ensure proper
			logging.


			Parameters:
			-----------
			job: hpbandster.distributed.dispatcher.Job object
				contains all the info about the run
		"""

		super().new_result(job)

		if job.result is None:
			# One could skip crashed results, but we decided to
			# assign a +inf loss and count them as bad configurations
			loss = np.inf
		else:
			# same for non numeric losses.
			# Note that this means losses of minus infinity will count as bad!
			loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf

		budget = job.kwargs["budget"]

		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []

		conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])

		self.randomforest.fit([conf.get_array()], [loss])

		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(loss)

		# update probs for the categorical parameters for later sampling
		self.logger.debug('done building a new model for budget %f, loss: %.2f'%(budget, loss))