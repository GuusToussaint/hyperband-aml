import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import numpy as np

def randomforest_randomsearch_BOHB_plot(problem_n, randomforest_runs, randomsearch_runs, bohb_runs, get_loss_from_run_fn=lambda r: r.loss, cmap=plt.get_cmap("tab10"), show=False):
    data = {}

    for i in ["randomforest", "randomsearch", "BOHB"]:
        data[i] = []

    for r in randomforest_runs:
        if r.loss is None:
            continue
        t = r.time_stamps['finished']
        l = get_loss_from_run_fn(r)
        data["randomforest"].append((t, l))

    for r in randomsearch_runs:
        if r.loss is None:
            continue
        t = r.time_stamps['finished']
        l = get_loss_from_run_fn(r)
        data["randomsearch"].append((t, l))

    for r in bohb_runs:
        if r.loss is None:
            continue
        t = r.time_stamps['finished']
        l = get_loss_from_run_fn(r)
        data["BOHB"].append((t, l))

    for i in ["randomforest", "randomsearch", "BOHB"]:
        data[i].sort()

    fig, ax = plt.subplots()

    for i, j in enumerate(["randomforest", "randomsearch", "BOHB"]):
        data[j] = np.array(data[j])
        ax.scatter(data[j][:, 0], data[j][:, 1], color=cmap(i), label=j)

        ax.step(data[j][:, 0], np.minimum.accumulate(
            data[j][:, 1]), where='post')

    ax.set_title('Losses for different budgets over time for problem {0}'.format(problem_n))
    ax.set_xlabel('wall clock time [s]')
    ax.set_ylabel('loss')
    ax.legend()
    if show:
        plt.show()

    return(fig, ax)

def normalized_plot(problems, get_loss_from_run_fn=lambda r: r.loss, cmap=plt.get_cmap("tab10")):
    normalized_data = {}
    loss_ranks = {}
    best_normalized_losses = {}
    longest_normalized_times = {}
    bohbrf_time = 0
    for i in ["randomforest", "randomsearch", "BOHB"]:
        normalized_data[i] = []
        loss_ranks[i] = []
        best_normalized_losses[i] = []
        longest_normalized_times[i] = []

    for problem in problems:
        # print(problem)
        min_losses = {}
        max_loss = 0
        max_time = 0
        

        for i in ["randomforest", "randomsearch", "BOHB"]:
            min_losses[i] = 999999
            result = hpres.logged_results_to_HBS_result('./results/results{0}{1}'.format(problem, i))
            all_runs = result.get_all_runs()
            for r in all_runs:
                if r.loss is None:
                    continue
                t = r.time_stamps['finished']
                l = get_loss_from_run_fn(r)
                if l > max_loss:
                    max_loss = l
                if t > max_time:
                    max_time = t
                if l < min_losses[i]:
                    min_losses[i] = l
            
            if i == 'randomforest':
                bohbrf_time += max_time

        ranks = sorted(min_losses, key=min_losses.get)
        for rnk, m in enumerate(ranks):
            loss_ranks[m].append(rnk+1)

        for i in ["randomforest", "randomsearch", "BOHB"]:
            min_normalized_loss = 2
            max_norm_loss = 0
            result = hpres.logged_results_to_HBS_result('./results/results{0}{1}'.format(problem, i))
            all_runs = result.get_all_runs()
            for r in all_runs:
                if r.loss is None:
                    continue
                t = r.time_stamps['finished']
                l = get_loss_from_run_fn(r)
                normalized_data[i].append((t / max_time, l / max_loss))

                if (l / max_loss) < min_normalized_loss:
                    min_normalized_loss = l / max_loss

                if (t / max_time) > max_norm_loss:
                    max_norm_loss = t / max_time
            best_normalized_losses[i].append(min_normalized_loss)
            longest_normalized_times[i].append(max_norm_loss)

    fig, ax = plt.subplots()

    for i, j in enumerate(["randomsearch", "randomforest", "BOHB"]):
        normalized_data[j] = np.array(normalized_data[j])
        ax.scatter(normalized_data[j][:, 0], normalized_data[j][:, 1], color=cmap(i), label=j, alpha=0.3)

        # ax.step(data[j][:, 0], np.minimum.accumulate(
        #     data[j][:, 1]), where='post')

    # Scatter normalized
    ax.set_title('Normalized loss over normalized time for all runs')
    ax.set_xlabel('wall clock time (normalized per problem)')
    ax.set_ylabel('loss (normalized per problem)')
    ax.legend()
    plt.show()


    # Boxplot all
    randomforest_data = np.array(normalized_data['randomforest'])
    randomsearch_data = np.array(normalized_data['randomsearch'])
    bohb_data = np.array(normalized_data['BOHB'])

    boxplot_time = [randomsearch_data[:, 0], randomforest_data[:, 0], bohb_data[:, 0]]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_time, labels=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized wall clock time for all runs')
    ax.set_xlabel('Method')
    ax.set_ylabel('Wall clock time (normalized per problem)')
    plt.show()

    boxplot_loss = [randomsearch_data[:, 1], randomforest_data[:, 1], bohb_data[:, 1]]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_loss, labels=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized loss for all runs')
    ax.set_xlabel('Method')
    ax.set_ylabel('Loss (normalized per problem)')
    plt.show()



    # Boxplot best and longest
    boxplot_time_longest = [longest_normalized_times['randomsearch'], longest_normalized_times['randomforest'], longest_normalized_times['BOHB']]
    fig, ax = plt.subplots()

    ax.boxplot(boxplot_time_longest, labels=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized longest wall clock time')
    ax.set_xlabel('Method')
    ax.set_ylabel('Wall clock time (normalized per problem)')
    plt.show()

    boxplot_loss_best = [best_normalized_losses['randomsearch'], best_normalized_losses['randomforest'], best_normalized_losses['BOHB']]
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_loss_best, labels=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized best loss')
    ax.set_xlabel('Method')
    ax.set_ylabel('Loss (normalized per problem)')
    plt.show()


    # Histogram all
    fig, ax = plt.subplots()
    ax.hist(boxplot_loss, bins=10, density=False, histtype='bar', label=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized loss for all runs')
    ax.set_xlabel('Loss (normalized per problem)')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(boxplot_time, bins=10, density=False, histtype='bar', label=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized wall clock time for all runs')
    ax.set_xlabel('Wall clock time (normalized per problem)')
    ax.legend()
    plt.show()

    #Histogram best and longest
    fig, ax = plt.subplots()
    ax.hist(boxplot_loss_best, bins=10, density=False, histtype='bar', label=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized best loss')
    ax.set_xlabel('Loss (normalized per problem)')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(boxplot_time_longest, bins=10, density=False, histtype='bar', label=["Randomsearch", "BOHB-RF", "BOHB"])
    ax.set_title('Normalized longest wall clock time')
    ax.set_xlabel('Wall clock time (normalized per problem)')
    ax.legend()
    plt.show()


    # Rank plot
    fig, ax = plt.subplots()
    ranks = np.arange(3)
    bar_width = 0.9/len(ranks)

    # print(np.bincount(loss_ranks['randomsearch'])[1:])
    ax.bar(ranks, np.bincount(loss_ranks['randomsearch'])[1:], bar_width, label='Random search')
    ax.bar(ranks + bar_width, np.bincount(loss_ranks['randomforest'])[1:], bar_width, label='BOHB-RF')
    ax.bar(ranks + 2 * bar_width, np.bincount(loss_ranks['BOHB'])[1:], bar_width, label='BOHB')
    

    # Prints some stats
    plt.xlabel("Rank")
    plt.ylabel("Number of occurences")
    plt.xticks(ranks + bar_width, ('1', '2', '3'))
    plt.legend()
    plt.show()

    print("Average randomsearch normalized best loss:", np.mean(best_normalized_losses['randomsearch']))
    print("Average BOHB-RF normalized best loss:", np.mean(best_normalized_losses['randomforest']))
    print("Average BOHB normalized best loss:", np.mean(best_normalized_losses['BOHB']), "\n")

    # print("Average randomsearch normalized wall clock time:", np.mean(randomsearch_data[:, 0]))
    # print("Average BOHB-RF normalized wall clock time:", np.mean(randomforest_data[:, 0]))
    # print("Average BOHB normalized wall clock time:", np.mean(bohb_data[:, 0]), "\n")

    print("Average randomsearch normalized wall clock time:", np.mean(longest_normalized_times['randomsearch']))
    print("Average BOHB-RF normalized wall clock time:", np.mean(longest_normalized_times['randomforest']))
    print("Average BOHB normalized wall clock time:", np.mean(longest_normalized_times['BOHB']), "\n")

    print("Average randomsearch rank:", np.mean(loss_ranks['randomsearch']))
    print("Average BOHB-RF rank:", np.mean(loss_ranks["randomforest"]))
    print("Average BOHB rank:", np.mean(loss_ranks["BOHB"]))

    print("BOHB-RF duration in seconds:", bohbrf_time)

    return(fig, ax)

problems = [3, 6, 11, 12, 14, 16, 18, 20, 21, 22, 23, 28, 43, 45, 49, 53, 58, 219, 2074]
# print(len(problems))

normalized_plot(problems=problems)


# for problem in problems:
#     print(problem)
#     result = {}
#     all_runs = {}
#     for i in ["randomforest", "randomsearch", "BOHB"]:
#         result[i] = hpres.logged_results_to_HBS_result('./results/results{0}{1}'.format(problem, i))
#         all_runs[i] = result[i].get_all_runs()

#     randomforest_randomsearch_BOHB_plot(problem, all_runs['randomforest'], all_runs['randomsearch'], all_runs['BOHB'])

#     plt.show()