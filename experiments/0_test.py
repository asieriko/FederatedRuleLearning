import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))
from pathlib import Path
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
from FedRLS.metrics import metrics
from FedRLS.FedRLS import FedRLS
from FedRLS.ex_fuzzy_manager import parse_rule_base
from datasets.load_dataset import read_keel_dat
from datasets.generate_datasets import generate_local_agents, load_dataset, plot_data_distribution
from experiments.plots import plot_rules_clients, plot_rules_cent


def non_federated_rules(dataset,data_seed=1,n_gen=30,n_pop=50,nRules=15,nAnts=4,fz_type_studied=fs.FUZZY_SETS.t1,tolerance=0.001,runner=1,random_seed=23,**args):
    # ,**args: to avoid "unexpected keyword for parameters not needed in this function but which are compiled into model_parameters"
        n_gen = n_gen
        n_pop = n_pop

        class_names = np.unique(dataset[2])
        X_train = dataset[0]
        X_test = dataset[1]
        y_train = dataset[2]
        y_test = dataset[3]
        precomputed_partitions = utils.construct_partitions(X_train, fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=nRules,
            nAnts=nAnts,
            fuzzy_type=fz_type_studied, 
            verbose=False,
            tolerance=tolerance, 
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=runner)
        fl_classifier = model
        
        fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, random_state = random_seed)
        # str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test,
        #                                         plot_rules=True, print_rules=True, plot_partitions=True,
        #                                         return_rules=True)


        predicted = fl_classifier.predict(X_test, out_class_names=True)
        train_performance = metrics(predicted, y_test)
        
        train_performance["train_performance"] = fl_classifier.performance
        print("-- Non Federated Case -- ")
        print(fl_classifier.rule_base)
        print(f"LOG:,NonFederated,{train_performance}")

        return fl_classifier.rule_base, train_performance

def main(datasetName, model_parameters, data_parameters, exp_name="Experiment", n_rep=3):
    path = Path(f"results/{datasetName}")
    Path.mkdir(path, parents=True, exist_ok=True)
    dataset = read_keel_dat(datasetName)
    model_parameters["class_names"] = dataset.target_names

    
    rulz = []
    partition = data_parameters["partition"]
    alpha = data_parameters["alpha"]
    for i in range(n_rep):  # FIXME: Change random_seed each rep
        data_parameters["random_seed"] += i
        clients_datasets, cls_map, one_client = generate_local_agents(dataset, **data_parameters)
        plot_data_distribution(cls_map, fileName= path / f"data_dist_{exp_name}_{i}")

        score_mlp = MLP(one_client) 
        print(f"MLP Score: {score_mlp}")
        # pre_scores, post_scores = MLP_federated(clients_datasets)
        # print(f"Federated MLP: {pre_scores}, {post_scores}")
        # print(f"Federated MLP: {np.average(pre_scores)}, {np.average(post_scores)}")  # It's a dict

        rules_cent, acc_cent = non_federated_rules(one_client, **model_parameters)
        plot_rules_cent(rules_cent,fileName=path / f"Rules_cent_{exp_name}_{i}",n_classes=len(dataset.target_names))
        # TODO: Create a plot for the not federated case, only rules and classes
        # print(clients_datasets)

        fedrls = FedRLS(clients_datasets, model_parameters)
        rules1, results = fedrls.local()  # rules1 is a dict, toomuch information in raw
        rules = fedrls.clients_rules()    # This returns a MasterRuleBase
        rulz.append(rules)
        for res in results:
            if res["type"] == "train":
                continue
            t = str(res["epoch"]) + "\t" + res["type"]
            for c in res["results"]:
                t += f"\t {c['final_accuracy']['accuracy']:.4f}"
            # t += "\n"  # Not necessary for print
            print(t)
         

        print(f"RULES \t cent \t \t {acc_cent['accuracy']:.4f} \t {acc_cent['f1']:.4f} \t {acc_cent['jaccard']:.4f}")
        print(f"MPL \t cent \t  \t {score_mlp['accuracy']:.4f} \t {score_mlp['f1']:.4f} \t {score_mlp['jaccard']:.4f}")
        # print(f"MPL \t fedavg \t  {np.average([ps['accuracy'] for ps in post_scores]):.4f} \t {np.average([ps['f1'] for ps in post_scores]):.4f} \t {np.average([ps['jaccard'] for ps in post_scores]):.4f}")


        plot_rules_clients(rules, fileName=path/f"Rules_clients_{exp_name}_{i}",n_classes=len(dataset.target_names))

        print(f"epoch \t type \t \t acc. \t f1s. \t jacc.")
        with open(Path("results") / "summary_fedrules.csv","a") as results_file:
            for i, res in enumerate(results):
                print("==========")
                print([c['initial_accuracy']['accuracy'] for c in res["results"]])
                print([c['final_accuracy']['accuracy'] for c in res["results"]])
                print("==========")
                a = np.average([c['final_accuracy']['accuracy'] for c in res["results"]])
                f = np.average([c['final_accuracy']['f1'] for c in res["results"]])
                j = np.average([c['final_accuracy']['jaccard'] for c in res["results"]])
                # results_file.write("datasetName,partition,alpha,random_seed,rep,type,accuracy,f1-score,jaccard\n")
                results_file.write(f"{datasetName},{partition},{alpha},{data_parameters['random_seed']},{i},{res['type']},{a},{f},{j}\n")
                if res["type"] == "update":
                    print(f"{res['epoch']} \t {res['type']} \t {a:.4f} \t {f:.4f} \t {j:.4f}")

        am = np.max([[c['final_accuracy']['accuracy'] for c in res["results"]] for res in results])
        mx_id = np.argmax([np.average([client['final_accuracy']["accuracy"] for client in res['results']]) for res in results])
        mx_acc = np.average([client['final_accuracy']["accuracy"] for client in results[mx_id]['results']])
        mx_f1 = np.average([client['final_accuracy']["f1"] for client in results[mx_id]['results']])
        mx_jacc = np.average([client['final_accuracy']["jaccard"] for client in results[mx_id]['results']])

        with open(Path("results") / "summary_all.csv","a") as results_file:
            results_file.write(f"{datasetName},{partition},{alpha},{data_parameters['random_seed']},{acc_cent['accuracy']:.4f},{acc_cent['f1']:.4f},{acc_cent['jaccard']:.4f},{score_mlp['accuracy']:.4f},{score_mlp['f1']:.4f},{score_mlp['jaccard']:.4f},,,,{mx_acc:.4f},{mx_f1:.4f},{mx_jacc:.4f},{mx_id}\n")
            # results_file.write(f"{datasetName},{partition},{alpha},{data_parameters['random_seed']},{acc_cent['accuracy']:.4f},{acc_cent['f1']:.4f},{acc_cent['jaccard']:.4f},{score_mlp['accuracy']:.4f},{score_mlp['f1']:.4f},{score_mlp['jaccard']:.4f},{np.average([ps['accuracy'] for ps in post_scores]):.4f},{np.average([ps['f1'] for ps in post_scores]):.4f},{np.average([ps['jaccard'] for ps in post_scores]):.4f},{mx_acc:.4f},{mx_f1:.4f},{mx_jacc:.4f},{mx_id}\n")

        # Warning: evolutionary_fit.py 505
        # rule_list = [[] for _ in range(self.n_classes)]  # FIXME: it is different the y send to the constructor and to this method
        # rule_list = [[] for _ in range(len(diff_consequents))]

if __name__ == "__main__":
    from FedAvg.mpl_skl import MLP_federated, MLP

    model_parameters = {
        "n_gen":200, #30
        "n_pop":100, #50
        "nRules":15,
        "nAnts":4,
        "fz_type_studied":fs.FUZZY_SETS.t1,
        "tolerance":0.001,
        "runner":1,
        "random_seed":23,
        "sim_threshold":0.7,
        "contradictory_factor":0.8,
        "max_retrains": 1,
        "class_names": None,
    }

    data_parameters = {
        "partition":"homo",
        "random_seed":23,
        "test_size":0.25, # Only for hetero-dir
        "alpha":0.5,  # Only for hetero-dir
        "min_partition_ratio":0.75, # Only for hetero-dir 1 -> n/n_clients, and 0.x fraction of that
        "n_clients": 3,
    }
    ds = ['adult','balance','coil2000','dermatology','german','ionosphere','lymphography','mammographic','nursery','penbased','satimage','spambase','thyroid','wdbc','wisconsin','appendicitis','car','crx','flare','heart','iris','monk-2','optdigits','phoneme','ring','segment','spectfheart','titanic','wine','australian','chess','housevotes','mushroom','page-blocks','pima','saheart','shuttle','splice','twonorm']
    # Fallos nursery - thyroid - spectfheart
    # ds = ['adult','balance','coil2000','dermatology','german','ionosphere','lymphography','mammographic','penbased','satimage','spambase','wdbc','wisconsin','appendicitis','car','crx','flare','heart','iris','monk-2','optdigits','phoneme','ring','segment','titanic','wine','australian','chess','housevotes','mushroom','page-blocks','pima','saheart','shuttle','splice','twonorm']
    ds = ['iris']
    for datasetName in ds:
        print(datasetName)
        n_rep = 5
        main(datasetName, model_parameters, data_parameters,exp_name=f"hetero-equal-{datasetName}",n_rep=n_rep)
        try:
            main(datasetName, model_parameters, data_parameters,exp_name=f"hetero-equal-{datasetName}",n_rep=n_rep)
        except Exception as inst:
            print(f"Error: {datasetName=}")
            print(type(inst))    # the exception type
            print(inst.args)     # arguments stored in .args
            print(inst)
            with open("errors.txt","a") as err_file:
                err_file.write(f"Error: {datasetName=}: {type(inst)} -> {inst.args} = {inst}")
