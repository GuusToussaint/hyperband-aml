import os
import pickle
import argparse
import netifaces

import hpbandster.core.nameserver as hpns   
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from hpbandster.optimizers import RandomSearch
from hpbandster.optimizers import RandomForestEI

import openml

from AML_project.openml_worker import PyTorchWorker as ptworker

parser = argparse.ArgumentParser(description='Example 5 - CNN on MNIST')
parser.add_argument('--min_budget',   type=float,
                    help='Minimum number of epochs for training.',    default=1)
parser.add_argument('--max_budget',   type=float,
                    help='Maximum number of epochs for training.',    default=27)
parser.add_argument('--n_iterations', type=int,
                    help='Number of iterations performed by the optimizer', default=8)
parser.add_argument(
    '--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str,
                    help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name', type=str,
                    help='Which network interface to use for communication.', default=netifaces.interfaces()[-2])
parser.add_argument('--shared_directory', type=str,
                    help='A directory that is accessible for all processes, e.g. a NFS share.', default='./results/results')
# parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='keras')

args = parser.parse_args()

task_start = 0
task_end = 1

host = '127.0.0.1'

benchmark_suite = openml.study.get_suite(
    'OpenML100')  # obtain the benchmark suite

for i in range(task_start, task_end):
    task_id = benchmark_suite.tasks[i]
    print("\n\nRunning task", task_id, "\n\n")
    task = openml.tasks.get_task(task_id)  # download the OpenML task

    bohb_logger = hpres.json_result_logger(
        directory=args.shared_directory + str(task_id) + "BOHB", overwrite=True)
    random_search_logger = hpres.json_result_logger(
        directory=args.shared_directory + str(task_id) + "randomsearch", overwrite=True)
    random_forest_logger = hpres.json_result_logger(
        directory=args.shared_directory + str(task_id) + "randomforest", overwrite=True)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host,
                         port=0, working_directory=args.shared_directory)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = ptworker(run_id=args.run_id, host=host, nameserver=ns_host,
                 nameserver_port=ns_port, timeout=120, task=task)
    w.run(background=True)


    # Run randomforest
    print("Start randomforest")
    randomforest = RandomForestEI(configspace=ptworker.get_configspace(),
                                run_id=args.run_id,
                                host=host,
                                nameserver=ns_host,
                                nameserver_port=ns_port,
                                result_logger=random_forest_logger,
                                min_budget=args.min_budget, max_budget=args.max_budget)
    res = randomforest.run(n_iterations=args.n_iterations)
    with open(os.path.join(args.shared_directory, 'results_randomforest.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    randomforest.shutdown()


    # Run randomsearch
    print("Start randomsearch")
    randomsearch = RandomSearch(configspace=ptworker.get_configspace(),
                                run_id=args.run_id,
                                host=host,
                                nameserver=ns_host,
                                nameserver_port=ns_port,
                                result_logger=random_search_logger,
                                min_budget=args.min_budget, max_budget=args.max_budget)
    res = randomsearch.run(n_iterations=args.n_iterations)

    with open(os.path.join(args.shared_directory, 'results_randomsearch.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    randomsearch.shutdown()


    # Run BOHB
    print("Start BOHB")
    bohb = BOHB(configspace=ptworker.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=bohb_logger,
                min_budget=args.min_budget, max_budget=args.max_budget
                )
    res = bohb.run(n_iterations=args.n_iterations)

    # store results
    with open(os.path.join(args.shared_directory, 'results_bohb.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown BOHB
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()