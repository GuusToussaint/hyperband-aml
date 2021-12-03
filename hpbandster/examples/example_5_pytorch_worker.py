"""
MNIST worker
Code based on example 5 of quickstart hpbandster

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Batch size              | integer        | [8, 256]    | logarithmically varied |
|                         |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            | float          | [0, 0.5]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Weight decay			  | float          | [0, 0.185]      |                        |
+-------------------------+----------------+-----------------+------------------------+
| Number of layers        | integer        | [1, 5]          | can only take integer  |
|                         |                |                 | values 1 to 5      |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in an FC layer          |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+


Set hyperparameters
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam}          |                        |
+-------------------------+----------------+-----------------+------------------------+

"""

try:
	import torch
	import torch.utils.data
	import torch.nn as nn
	import torch.nn.functional as F
except:
	raise ImportError("For this example you need to install pytorch.")

try:
	import torchvision
	import torchvision.transforms as transforms
except:
	raise ImportError("For this example you need to install pytorch-vision.")



import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)



class PyTorchWorker(Worker):
	def __init__(self, N_train = 8192, N_valid = 1024, **kwargs):
		super().__init__(**kwargs)

		# Load the MNIST Data here
		self.train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
		self.test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())
		
		self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train))
		self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train, N_train+N_valid))

	def compute(self, config, budget, working_directory, *args, **kwargs):
		"""
		Simple example for a compute function using a feed forward network.
		It is trained on the MNIST dataset.
		The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
		"""
		self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=config['batch_size'], sampler=self.train_sampler)
		self.validation_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=config['batch_size'], sampler=self.validation_sampler)

		self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=config['batch_size'], shuffle=False)

		# device = torch.device('cpu')
		model = FCnet(num_layers=config['num_layers'],
							dropout_rate=config['dropout_rate'],
							num_fc_units=config['num_fc_units'],
							kernel_size=3
		)

		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

		for epoch in range(int(budget)):
			loss = 0
			model.train()
			for i, (x, y) in enumerate(self.train_loader):
				optimizer.zero_grad()
				output = model(x)
				loss = F.nll_loss(output, y)
				loss.backward()
				optimizer.step()

		train_accuracy = self.evaluate_accuracy(model, self.train_loader)
		validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
		test_accuracy = self.evaluate_accuracy(model, self.test_loader)

		return ({
			'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
			'info': {	'test accuracy': test_accuracy,
						'train accuracy': train_accuracy,
						'validation accuracy': validation_accuracy,
						'number of parameters': model.number_of_parameters(),
					}
		})

	def evaluate_accuracy(self, model, data_loader):
		model.eval()
		correct=0
		with torch.no_grad():
			for x, y in data_loader:
				output = model(x)
				#test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(y.view_as(pred)).sum().item()
		#import pdb; pdb.set_trace()	
		accuracy = correct/len(data_loader.sampler)
		return(accuracy)


	@staticmethod
	def get_configspace():
		"""
		It builds the configuration space with the needed hyperparameters.
		It is easily possible to implement different types of hyperparameters.
		Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
		:return: ConfigurationsSpace-Object
		"""
		cs = CS.ConfigurationSpace()

		lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-2, default_value=1e-3, log=True)
		batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=8, upper=256, default_value=16, log=True)
		droprate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.5, default_value=0)
		weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=0, upper=0.185, default_value=0)
		n_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=1, upper=5, default_value=1)
		n_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=8, log=True)

		cs.add_hyperparameters([lr, batch_size, droprate, weight_decay, n_layers,
		n_fc_units])

		return cs




class FCnet(torch.nn.Module):
	def __init__(self, num_layers, dropout_rate, num_fc_units, kernel_size):
		super().__init__()
		
		self.conv = nn.Conv2d(1, 64, kernel_size=kernel_size)
		output_size = (28-kernel_size+1)//2
		self.conv_output_size = 64*output_size*output_size
		self.layers = [None]*num_layers
		

		if num_layers != 1:
			self.layers[0] = nn.Linear(self.conv_output_size, num_fc_units)
			if num_layers > 2:
				for i in range(1, num_layers-1):
					self.layers[i] = nn.Linear(num_fc_units, num_fc_units)

			self.layers[-1] = nn.Linear(num_fc_units, 10)

		else:
			self.layers[0] = nn.Linear(self.conv_output_size, 10)

		self.dropout = nn.Dropout(p = dropout_rate)
		
		output_size = (28-kernel_size + 1)//2


	def forward(self, x):
		
		# switched order of pooling and relu compared to the original example
		# to make it identical to the keras worker
		# seems to also give better accuracies
		x = F.max_pool2d(F.relu(self.conv(x)), 2)

		x = self.dropout(x)

		x = x.view(-1, self.conv_output_size)

		if len(self.layers) != 1:
			for i in range(len(self.layers)-1):
				x = self.layers[i](x)
				x = F.relu(x)
				x = self.dropout(x)

		x = self.layers[-1](x)

		return F.log_softmax(x, dim=1)


	def number_of_parameters(self):
		return(sum(p.numel() for p in self.parameters() if p.requires_grad))



if __name__ == "__main__":
	worker = PyTorchWorker(run_id='0')
	cs = worker.get_configspace()
	
	config = cs.sample_configuration().get_dictionary()
	print(config)
	res = worker.compute(config=config, budget=2, working_directory='.')
	print(res)
