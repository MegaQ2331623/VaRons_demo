from utils import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Parameters of experiments")
    parser.add_argument('-r','--repeat', default='20',type=int,required=False,help="Number of repeated trials")
    parser.add_argument('-re', '--retrain', default=False,type=bool,required=False,help="Plot with exist results without training if true, retrain otherwise.")
    parser.add_argument('-t', '--T', default=1000,type=int,required=False,help="Total time horizon.")
    parser.add_argument('-K', '--K', default=10,type=int,required=False,help="Number of actions.")
    parser.add_argument('-env', '--environment', default='drift',type=str,required=False,help="Variation environment, options = 'drift' or 'random'.")
    args = parser.parse_args()
    data,d,m,drift_rep,K,T,rep,new_trial,env,algo_list,contextual_features = get_para(args)
    dataset='OMS'
    for algo in algo_list:
        result_path = './results/'+str(dataset)+'_'+str(K)+'_'+env+'_'+algo+'.npy'
        result_not_find = not os.path.exists(result_path)
        if new_trial or result_not_find:
            if new_trial:
                print ('Retrain = True, retraining for algorithm ',algo)
            elif not new_trial and result_not_find:
                print ('Retrain = False, but there is no corresponding results, retraining for algorithm ',algo)
            cum_regret_hist_li=[]

            for repeat in range(rep):
                X_test, y_test, action_set, models = data
                m = find_gt_pf(K, d, X_test, y_test, models)
                alg,weight_sequence=trial_init(env,algo,K,m,T,d,drift_rep)

                for t in range(T):
                    w_t=weight_sequence[t]
                    action_set = get_actionset(data,dataset,K)
                    a_t = alg.select_action(w_t,contextual_features,action_set)
                    main_objective, inst_regret = cal_inst_regret(data,dataset,w_t,K,d,a_t,action_set)
                    alg.update(w_t,contextual_features,a_t,main_objective,inst_regret)
                cum_regret,cum_regret_hist=alg.fetch_regret()
                cum_regret_hist_li.append(cum_regret_hist)
            cum_regret_hist_li=np.array(cum_regret_hist_li)
            np.save(result_path, cum_regret_hist_li)

        results = np.load(result_path)
        plt.plot(range(len(results[0])), results.mean(axis=0), label=algo)
    if not new_trial and not result_not_find:
        print('Retrain = False and results exist, skip to ploting.')
    # Add labels and title
    plt.xlabel('Time steps')
    plt.ylabel('Average cumulative regret')
    plt.title('Cumulative regret on K='+str(K)+', environment='+str(env))

    # Add a legend (if needed)
    plt.legend()

    # Show the plot
    plt.show()



if __name__ == '__main__':
    main()
