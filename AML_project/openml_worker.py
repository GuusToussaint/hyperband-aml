import torch
import numpy as np

from AML_project.simpledataset import SimpleDataset
from hpbandster.core.worker import Worker

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from AML_project.fcnet import FCnet

class PyTorchWorker(Worker):
    def __init__(self, task, **kwargs):
        super().__init__(**kwargs)

        X, y = task.get_X_and_y()
        dataset = SimpleDataset(X, y)
        train_val_size = int(len(dataset)*0.8)
        train_size = int(train_val_size * 0.75)
        val_size = train_val_size - train_size
        test_size = len(dataset) - train_val_size

        self.train_val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, (train_val_size, test_size))
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        #     train_val_dataset, (train_size, val_size))

        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            range(train_size))
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            range(train_size, train_val_size))

        # self.in_len = X.shape
        # self.out_shape = y.shape
        if len(y.shape) == 1:
            self.out_shape = 1
        else:
            self.out_shape = y.shape[1]
        self.in_shape = 1
        # for i in X.shape:
        #     self.in_shape *= i
        self.in_shape = X.shape[1]
        self.X = X
        self.y = y

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_val_dataset, batch_size=config['batch_size'], sampler=self.train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=self.train_val_dataset, batch_size=config['batch_size'], sampler=self.validation_sampler)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=config['batch_size'], shuffle=False)

        # print("Lens", len(self.train_dataset), len(self.train_sampler), len(self.train_loader), len(self.val_dataset), len(self.validation_sampler), len(self.validation_loader))

        # device = torch.device('cpu')
        model = FCnet(num_layers=config['num_layers'],
                      dropout_rate=config['dropout_rate'],
                      num_fc_units=config['num_fc_units'],
                      kernel_size=3,
                      out_size=self.out_shape,
                      in_size=self.in_shape
                      )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['lr'], weight_decay=config['weight_decay'])

        for epoch in range(int(budget)):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(x).squeeze().to(torch.float32)
                y = y.to(torch.float32)
                # print(type(output), type(y))
                loss = criterion(output, y)
                # print("Epoch:", epoch, "\tloss:", loss.item())
                loss.backward()
                optimizer.step()

        train_loss = self.evaluate_loss(model, self.train_loader)
        validation_loss = self.evaluate_loss(model, self.validation_loader)
        test_loss = self.evaluate_loss(model, self.test_loader)
        return ({
            'loss': validation_loss,  # remember: HpBandSter always minimizes!
            'info': {'test loss': test_loss,
                     'train loss': train_loss,
                     'validation loss': validation_loss,
                     'number of parameters': model.number_of_parameters(),
                     }
        })

        # train_accuracy = self.evaluate_accuracy(model, self.train_loader)
        # validation_accuracy = self.evaluate_accuracy(
        #     model, self.validation_loader)
        # test_accuracy = self.evaluate_accuracy(model, self.test_loader)

        # return ({
        #         'loss': 1-validation_accuracy,  # remember: HpBandSter always minimizes!
        #         'info': {'test accuracy': test_accuracy,
        #                  'train accuracy': train_accuracy,
        #                  'validation accuracy': validation_accuracy,
        #                  'number of parameters': model.number_of_parameters(),
        #                  }
        #         })

    def evaluate_loss(self, model, data_loader):
        # print("data_loader length", len(data_loader), "\n", data_loader)
        model.eval()
        with torch.no_grad():
            loss = []
            f = torch.nn.MSELoss()
            for i, (x, y) in enumerate(data_loader):
                output = model(x).squeeze()
                # print(y, output, y.shape, output.shape, type(y), type(output))
                loss.append(f(output.to(torch.float32),
                            y.to(torch.float32)).item())
        return(float(np.mean(loss)))

    # def evaluate_accuracy(self, model, data_loader):
    #     model.eval()
    #     correct = 0
    #     with torch.no_grad():
    #         for x, y in data_loader:
    #             output = model(x)
    #             # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
    #             # get the index of the max log-probability
    #             pred = output.max(1, keepdim=True)[1]
    #             correct += pred.eq(y.view_as(pred)).sum().item()
    #     #import pdb; pdb.set_trace()
    #     accuracy = correct/len(data_loader.sampler)
    #     return(accuracy)

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-2, default_value=1e-3, log=True)
        batch_size = CSH.UniformIntegerHyperparameter(
            'batch_size', lower=8, upper=256, default_value=16, log=True)
        droprate = CSH.UniformFloatHyperparameter(
            'dropout_rate', lower=0, upper=0.5, default_value=0)
        weight_decay = CSH.UniformFloatHyperparameter(
            'weight_decay', lower=0, upper=0.185, default_value=0)
        n_layers = CSH.UniformIntegerHyperparameter(
            'num_layers', lower=1, upper=5, default_value=1)
        n_fc_units = CSH.UniformIntegerHyperparameter(
            'num_fc_units', lower=8, upper=256, default_value=8, log=True)

        cs.add_hyperparameters([lr, batch_size, droprate, weight_decay, n_layers,
                                n_fc_units])

        return cs