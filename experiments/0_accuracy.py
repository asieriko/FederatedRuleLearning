import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas  as pd
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
from FedRLS.metrics import metrics
from FedRLS.FedRLS import FedRLS
from FedRLS.utils import  average_dict
from datasets.load_dataset import read_json_dataset
from datasets.generate_datasets import generate_one_client
from experiments.plots import plot_rules_clients, plot_rules_cent, plot_metrics
from FedRLS.models import FRBCmodel, FedRLSmodel

def non_federated_rules(dataset,model_parameters,fp=None):

        class_names = np.unique(dataset[2])
        X_train = dataset[0]
        X_test = dataset[1]
        y_train = dataset[2]
        y_test = dataset[3]
        if fp:
            precomputed_partitions = fp
        else:
            # NOTE: The construction of partitions considers quantiles, so I pass only
            # max and min like the federated case
            X_range = np.stack([np.min(X_train, axis=0), np.max(X_train, axis=0)])
            precomputed_partitions = utils.construct_partitions(X_range, model_parameters.fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=model_parameters.nRules,
            nAnts=model_parameters.nAnts,
            fuzzy_type=model_parameters.fz_type_studied,
            verbose=model_parameters.verbose,
            tolerance=model_parameters.tolerance,
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=model_parameters.runner)
        fl_classifier = model

        fl_classifier.fit(X_train, y_train, n_gen=model_parameters.n_gen, pop_size=model_parameters.n_pop, random_state = model_parameters.random_seed)
        # str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test,
        #                                         plot_rules=True, print_rules=True, plot_partitions=True,
        #                                         return_rules=True)


        predicted = fl_classifier.predict(X_test, out_class_names=True)
        train_performance = metrics(predicted, y_test)

        train_performance["train_performance"] = fl_classifier.performance
        return fl_classifier.rule_base, train_performance


def evaluate_otherdata(datasets, X_train, rule_base, model_parameters,fp=None):
    if fp:
        precomputed_partitions = fp
    else:
        precomputed_partitions = utils.construct_partitions(X_train, model_parameters.fz_type_studied)
    model = GA.BaseFuzzyRulesClassifier(
        nRules=model_parameters.nRules,
        nAnts=model_parameters.nAnts,
        fuzzy_type=model_parameters.fz_type_studied,
        verbose=False,
        tolerance=model_parameters.tolerance,
        linguistic_variables=precomputed_partitions,
        runner=model_parameters.runner)
    fl_classifier = model
    fl_classifier.rule_base=rule_base

    results = []
    for dataset in datasets:
        X_test = dataset[1]
        y_test = dataset[3]
        predicted = fl_classifier.predict(X_test, out_class_names=True)
        performance = metrics(predicted, y_test)
        results.append(performance)

    return average_dict(results)

def main(datasetName, model_parameters, fedrls_parameters, nclients, alpha, exp_name="Experiment", n_rep=3):
    path = Path(f"results/{datasetName}")
    Path.mkdir(path, parents=True, exist_ok=True)

    rulz = []
    all_results = []
    headers_results = ["Dataset","NClients","Alpha", "Repetition","Method","N_rules", "Accuracy"]
    for i in range(n_rep):
        print(f" fold {i+1}/{n_rep}")
        dataset_path = Path("datasets") / datasetName / f"{datasetName}_a{alpha}_n{nclients}_f{i+1}-{n_rep}.json"
        clients_datasets = read_json_dataset(datasetName, dataset_path)
        one_client = generate_one_client(clients_datasets)
        n_classes = len(np.unique(np.hstack((one_client[2],one_client[3]))))

        # --> MLP and FedAVG <--
        # score_mlp = MLP(one_client)
        # print(f"MLP Score: {score_mlp}")
        # pre_scores, post_scores = MLP_federated(clients_datasets)
        # print(f"Federated MLP: {pre_scores}, {post_scores}")
        # print(f"Federated MLP: {np.average(pre_scores)}, {np.average(post_scores)}")  # It's a dict


        # --> Centralized version <--
        print("-- Non Federated Case -- ")
        X_range = np.stack([np.min(one_client[0], axis=0), np.max(one_client[0], axis=0)])
        fp = utils.construct_partitions(X_range, model_parameters.fz_type_studied)
        rules_cent, acc_cent = non_federated_rules(one_client, model_parameters, fp=fp)
        plot_rules_cent(rules_cent,fileName=path / f"Rules_cent_{exp_name}_{i}.png",n_classes=n_classes)
        n_rules_cent = sum([len(r) for r in [rb.get_rulebase_matrix() for rb in rules_cent.rule_bases]])
        sco_cent = [sc for sc in [rb.get_scores().tolist() for rb in rules_cent.rule_bases]]
        nactiv_cent = sum([len(np.nonzero(b)[0]) for b in [a for a in sco_cent]])
        print(f"LOG:,NonFederated,{acc_cent['accuracy']}, {n_rules_cent} rules")
        all_results.append([i,"CL",n_rules_cent,acc_cent['accuracy']])

        # --> Distributed - non-collaborative <--
        # Study the case in which each client learns from its dataset, but
        # doesn't collaborate among them to create a unique model
        nc_avg = []
        nc_std = []


        nc_results = []
        print("-- Distributed Non Collaborative Case -- ")
        # TODO: Test if the data is the same for each client as the federated case
        # TODO: Try with the precomputed partitons of all data
        LL_Rules = []
        LL_neq_rules = []
        X_range = np.stack([np.min(one_client[0],axis=0),np.max(one_client[0],axis=0)])
        fp = utils.construct_partitions(X_range, model_parameters.fz_type_studied)
        for client_ds in clients_datasets:
            rul_i, acc_i = non_federated_rules(client_ds, model_parameters, fp=fp)
            acc_i_global = evaluate_otherdata(clients_datasets, client_ds[0], rul_i, model_parameters,fp=fp)
            nc_results.append([acc_i["accuracy"],acc_i["f1"],acc_i["jaccard"], acc_i_global["accuracy"],acc_i_global["f1"],acc_i_global["jaccard"]])

            for rb in rul_i.rule_bases:
                for r in rb.get_rulebase_matrix():
                    LL_neq_rules.append(r)  # FIXME: add consequent?
            LL_Rules.append(rul_i)
        nc_avg = np.average(np.array(nc_results), axis=0)
        nc_std = np.std(np.array(nc_results), axis=0)
        n_rules_nc = len({tuple(arr) for arr in np.array(LL_neq_rules)})
        print(f"LOG: AVG {nc_avg[0]}, {n_rules_nc} rules")
        print(f"LOG: AVG {nc_avg[3]}, Global")
        all_results.append([i,"LL",n_rules_nc,nc_avg[0]])
        all_results.append([i,"GLL",n_rules_nc,nc_avg[3]])
        # compute average for all runs... The avg of avgs? the std?
        # TODO: Create a plot for the not federated case, only rules and classes
        # print(clients_datasets)


        # --> Federated Version - adaptative <--
        print("-- Federated Adaptative Case -- ")
        adaptative_fedrls = fedrls_parameters._replace(adaptative=True)
        fedrls_afl = FedRLS(clients_datasets, model_parameters, adaptative_fedrls)
        rules_afl_dic, results_afl = fedrls_afl.local()  # rules_afl_dic is a dict, toomuch information in raw
        rules_afl = fedrls_afl.clients_rules()    # This returns a MasterRuleBase
        rulz.append(rules_afl)
        scores_afl = []
        for mrb in rules_afl:
            for rb in mrb:
                for r in rb:
                    scores_afl.append(r.score)
        n_rules_afl = len(scores_afl)/len(rules_afl)
        n_rules_activ_afl = len([1 for x in scores_afl if x > 0])/len(rules_afl)
        performances_afl = []
        for client in fedrls_afl.clients.values():
            performances_afl.append(client['agent'].eval_test())
        # Evaluate each client with all the test datasets
        results_gafl = fedrls_afl.eval_otherdata(clients_datasets)
        print(f"LOG: AVG {np.mean([x['final_accuracy']['accuracy'] for x in performances_afl])}, {n_rules_activ_afl} rules")
        print(f"LOG: AVG {results_gafl['accuracy']}, {n_rules_afl} rules (GAFL)")
        all_results.append([i,"AFL",n_rules_activ_afl,np.mean([x['final_accuracy']['accuracy'] for x in performances_afl])])
        all_results.append([i,"GAFL",n_rules_afl,results_gafl["accuracy"]])

        # --> Federated Version - Non - adaptative <--
        print("-- Federated Non-Adaptative Case -- ")
        non_adaptative_fedrls = fedrls_parameters._replace(adaptative=False)
        rulz_nafl = []
        fedrls_nafl = FedRLS(clients_datasets, model_parameters, non_adaptative_fedrls)
        rules_nafl_dic, results_nafl = fedrls_nafl.local()  # rules1 is a dict, toomuch information in raw
        rules_nafl = fedrls_nafl.clients_rules()  # This returns a MasterRuleBase
        scores_fed_nafl = []
        for mrb in rules_nafl:
            for rb in mrb:
                for r in rb:
                    scores_fed_nafl.append(r.score)
        n_rules_nafl = len(scores_fed_nafl)/len(rules_nafl)
        rulz_nafl.append(rules_nafl)
        performances_nafl = []
        for client in fedrls_nafl.clients.values():
            performances_nafl.append(client['agent'].eval_test())
        # Evaluate each client with all the test datasets
        results_gnafl = fedrls_nafl.eval_otherdata(clients_datasets)
        print(f"LOG: AVG {np.mean([x['final_accuracy']['accuracy'] for x in performances_nafl])}, {n_rules_nafl} rules")
        print(f"LOG: AVG {results_gnafl['accuracy']}, {n_rules_nafl} rules (GNAFL)")
        all_results.append([i,"NAFL",n_rules_nafl,np.mean([x['final_accuracy']['accuracy'] for x in performances_nafl])])
        all_results.append([i,"GNAFL", n_rules_nafl, results_gnafl["accuracy"]])


        plot_rules_clients(rules_afl, fileName=path/f"Rules_clients_{exp_name}_{i}.png",n_classes=n_classes)

        '''
        # Store in files
        with open(Path("results") / datasetName / f"summary_fedrules_{exp_name}.csv","a") as results_file:
            # Cenralized learning
            results_file.write(f"{datasetName},{nclients},{alpha},CL,{i},'',{n_rules_cent}, {nactiv_cent},{acc_cent['accuracy']}, {acc_cent['f1']}, {acc_cent['jaccard']}\n")
            # Local Leaarning non Collaborative
            results_file.write(f"{datasetName},{nclients},{alpha},LL,{i},'',{n_rules_nc},{n_rules_nc},{nc_avg[0]},{nc_avg[1]},{nc_avg[2]}\n")
            results_file.write(f"{datasetName},{nclients},{alpha},LLG,{i},'',{n_rules_nc},{n_rules_nc},{nc_avg[3]},{nc_avg[4]},{nc_avg[5]}\n")

            # Federated adaptative
            for i, res in enumerate(results_afl):
                a = np.average([c['final_accuracy']['accuracy'] for c in res["results"]])
                f = np.average([c['final_accuracy']['f1'] for c in res["results"]])
                j = np.average([c['final_accuracy']['jaccard'] for c in res["results"]])
                results_file.write(f"{datasetName},{nclients},{alpha},AFL,{i},{res['type']},{n_rules_afl}, {n_rules_activ_afl},{a},{f},{j}\n")

            # Federated non-adaptative
            for i, res in enumerate(results_nafl):
                a = np.average([c['final_accuracy']['accuracy'] for c in res["results"]])
                f = np.average([c['final_accuracy']['f1'] for c in res["results"]])
                j = np.average([c['final_accuracy']['jaccard'] for c in res["results"]])
                results_file.write(f"{datasetName},{nclients},{alpha},ANFL,{i},{res['type']},{n_rules_nafl},{n_rules_nafl},{a},{f},{j}\n")
            '''
        am = np.max([[c['final_accuracy']['accuracy'] for c in res["results"]] for res in results_afl])
        mx_id = np.argmax([np.average([client['final_accuracy']["accuracy"] for client in res['results']]) for res in results_afl])
        mx_acc = np.average([client['final_accuracy']["accuracy"] for client in results_afl[mx_id]['results']])
        mx_f1 = np.average([client['final_accuracy']["f1"] for client in results_afl[mx_id]['results']])
        mx_jacc = np.average([client['final_accuracy']["jaccard"] for client in results_afl[mx_id]['results']])

        with open(Path("results") / datasetName / f"summary_all_{exp_name}.csv","a") as results_file:
            results_file.write(f"{datasetName},{nclients},{alpha},{acc_cent['accuracy']:.4f},{acc_cent['f1']:.4f},{acc_cent['jaccard']:.4f},,,,,,{mx_acc:.4f},{mx_f1:.4f},{mx_jacc:.4f},{mx_id}\n")

        # Warning: evolutionary_fit.py 505
        # rule_list = [[] for _ in range(self.n_classes)]  # FIXME: it is different the y send to the constructor and to this method
        # rule_list = [[] for _ in range(len(diff_consequents))]

    df = pd.DataFrame(all_results,columns=["Fold","Method","Rules","Accuracy"])
    df["Dataset"] = datasetName
    df["NClients"] = nclients
    df["Alpha"] = alpha
    df["folds"] = n_rep
    return df


def pretty_print_pt(pt):
    # PRINT PT
    # Extract the mean values for Accuracy and Rules
    mean_accuracy = pt[('mean', 'Accuracy')]
    std_accuracy = pt[('std', 'Accuracy')]
    mean_rules = pt[('mean', 'Rules')]
    std_rules = pt[('std', 'Rules')]

    # Combine the results into a single DataFrame
    result = pd.DataFrame({
        'Method': mean_accuracy.index.get_level_values('Method'),
        'Mean Accuracy': mean_accuracy.values,
        'Std Accuracy': std_accuracy.values,
        'Mean Rules': mean_rules.values,
        'Std Rules': std_rules.values
    })
    custom_order = ['CL', 'LL', 'GLL', 'AFL', 'GAFL', 'NAFL', 'GNAFL']
    result['Method'] = pd.Categorical(result['Method'], categories=custom_order, ordered=True)
    result = result.sort_values(by="Method")

    # Print the results
    print(result)

if __name__ == "__main__":
    # from FedAvg.mpl_skl import MLP_federated, MLP

    model_parameters = FRBCmodel(
        n_gen =  30,  # 30
        n_pop =  50,  # 50
        nRules =  20,
        nAnts =  4,
        fz_type_studied =  fs.FUZZY_SETS.t1,
        tolerance =  0.001,
        runner =  1,
        random_seed =  23,
        class_names =  None,
    )

    fedrls_parameters = FedRLSmodel(
        sim_threshold=0.7,
        contradictory_factor=0.8,
        max_retrains=0,
        adaptative=False,
    )


    ds = ['adult','balance','coil2000','dermatology','german','ionosphere','lymphography','mammographic','nursery','penbased','satimage','spambase','thyroid','wdbc','wisconsin','appendicitis','car','crx','flare','heart','iris','monk-2','optdigits','phoneme','ring','segment','spectfheart','titanic','wine','australian','chess','housevotes','mushroom','page-blocks','pima','saheart','shuttle','splice','twonorm']
    ds_less_9_att = ['appendicitis', 'balance', 'banana', 'breast', 'bupa', 'car', 'contraceptive', 'ecoli', 'glass',
          'haberman', 'hayes-roth', 'iris', 'mammographic', 'monk-2', 'newthyroid', 'nursery',
          'phoneme', 'pima', 'post-operative', 'saheart', 'shuttle', 'tic-tac-toe', 'titanic', 'wisconsin']
    ds_less_9_att = ['appendicitis', 'balance',# 'banana', 'breast', 'bupa', 'car', 'contraceptive', 'ecoli', 'glass',
          # 'haberman', 'hayes-roth',
          'iris', 'mammographic', 'monk-2', #'newthyroid', 'nursery',
          'phoneme', 'pima', 'post-operative', 'saheart', #'shuttle',
          'tic-tac-toe', 'titanic', 'wisconsin']
    # ds_less_9_att += ['abalone',  'kr-vs-k', 'led7digit', 'yeast'] # More than 10 classes
    # filename => datasetName_n{CLIENTS}_a{ALPHA}_f{FOLDS}.json
    # CLIENTS = 5/10
    # ALPHA = 0.5/100 hetero/homo
    # FOLDS = 5

    if len(sys.argv) > 1:
        ds = [sys.argv[1]]


    n_clients = [10]#[3, 5, 10]
    alphas = [0.5, 500]
    folds = [5]#[5, 10]
    all_res = None
    ds = ['iris','balance']
    for datasetName in ds:
        for fold in folds:
            for nclients in n_clients:
                for alpha in alphas:
                    print(f"------ New Experiment: {datasetName}: {alpha=} and {nclients=} ------")
                    name = "10-12_20R-tst"
                    experiment = f'{datasetName}_n{nclients}_a{alpha}_{fold}CV'
                    # df = main(datasetName, model_parameters, fedrls_parameters, nclients, alpha, exp_name=f"-{experiment}-{name}",
                    #          n_rep=fold)
                    try:
                        df = main(datasetName, model_parameters, fedrls_parameters, nclients, alpha, exp_name=f"-{experiment}-{name}",n_rep=fold)
                        df = df[["Dataset","NClients","Alpha","Fold", "Method", "Rules", "Accuracy"]]

                        pt = df.pivot_table(values=["Accuracy", "Rules"],
                                            index=["Method", "Dataset", "Alpha", "NClients"],
                                            aggfunc=["count", "mean", "std"])

                        print(f"------ Summary: {datasetName}: {alpha=} and {nclients=} {fold}CV------")
                        plot_metrics(pt, "Accuracy",
                                methods_to_plot=['CL','LL','GLL','AFL','GAFL','NAFL','GNAFL'],
                                plot_title=f'Accuracy ({fold}CV) for the {datasetName} dataset\n{nclients} clients , $alpha$={alpha}',
                                file_name=Path("results") / datasetName /f"Accuracy_{experiment}-{name}.png")
                        plot_metrics(pt, "Rules",
                                methods_to_plot=['CL','LL','GLL','AFL','GAFL','NAFL','GNAFL'],
                                plot_title=f'Number of rules ({fold}CV) for the {datasetName} dataset\n{nclients} clients , $alpha$={alpha}',
                                file_name=Path("results") / datasetName /f"Rules_{experiment}-{name}.png")
                        pretty_print_pt(pt)
                        pt.columns = ['_'.join(col) for col in pt.columns]
                        # pt = pt.reset_index()
                        # pt.insert(2,"folds",fold)
                        # print(pt[["Method","mean_Accuracy","std_Accuracy","mean_Rules","std_Rules"]])

                        if all_res is None:
                            all_res = pt
                            # all_res.columns = list(map("_".join, all_res.columns))
                        else:
                            all_res = pd.concat([all_res,pt])
                        # pt = pt.droplevel([1, 2, 3, 4, 5])  # To keep only the method name
                        # pt.plot(kind="bar")
                        df.to_csv(Path("results") / datasetName /f"df_results_{experiment}-{name}_{fold}CV.csv", mode="a", header=False)
                        all_res.to_csv(Path("results") /f"df_pivot_results_partial-{name}.csv", header=True)
                    except Exception as inst:
                        print(f"### Error: {datasetName=} ###")
                        print(type(inst))    # the exception type
                        print(inst.args)     # arguments stored in .args
                        print(inst)
                        with open(f"errors-{name}.txt","a") as err_file:
                            err_file.write(f"Error: {datasetName=}: {type(inst)} -> {inst.args} = {inst}\n")
    all_res.to_csv(Path("results") / f"df_pivot_results_{name}.csv",  header=True) # Does not have folds information (5/10)
