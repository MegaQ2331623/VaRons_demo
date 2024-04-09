from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
import cvxpy as cp
import numpy as np
from sklearn.model_selection import train_test_split
import math
import random
import joblib
import os

def get_actionset(data,dataset,K):
    if dataset == 'SNW' or 'NOC':
        return random.sample(range(len(data[0])), K)

def cal_inst_regret(data,dataset,w_t,K,d,a_t,action_set):
    if dataset == 'OMS':
        X_test, y_test, action_set, models = data
        _, X_test_t, _, y_test_t = train_test_split(X_test, y_test, test_size=0.05)
        loss_vectors = fetch_loss_vectors(w_t, K, d, X_test_t, y_test_t, models)
        main_objective = loss_vectors[a_t]
        best_action = find_the_best_model(loss_vectors)
        inst_regret = main_objective - loss_vectors[best_action]
    elif dataset == 'SNW' or 'NOC':
        X, obj1, obj2 = data
        cur_obj = [obj1[a_t], obj2[a_t]]
        main_objective = np.dot(cur_obj, w_t)
        minimum_loss = find_the_best_action(obj1, obj2, w_t, action_set)
        inst_regret = main_objective - minimum_loss
    return main_objective, inst_regret

def get_para(args):
    new_trial = args.retrain
    rep = args.repeat
    T = args.T
    K = args.K
    env = args.environment
    # if dataset not in ['OMS', 'SNW','NOC']:
    #     raise ValueError('Experimental types must be selected within oms, snw or noc.')
    # if K>10 and dataset=='OMS':
    #     raise ValueError('Maximum 10 models are available for online model selection.')
    # if dataset =='OMS':
    d = 4  # number of sub-objectives
    m = 0
    drift_rep = 2
    X_test, y_test, action_set, models = global_init(K)
    data = [X_test, y_test, action_set, models]
    algo_list=['VaRons','EXP3','D-UCB','SW-UCB','GT_PF']
    contextual_features = None
    # elif dataset =='SNW' or 'NOC':
    #     d = 2  # number of sub-objectives
    #     X, obj1, obj2 = preprocessing(dataset)
    #     m = len(X[0])
    #     drift_rep = 10
    #     data = [X, obj1, obj2]
    #     algo_list = ['ConVaRons','RestartUCB','WLinUCB','SWLinUCB']
    #     contextual_features = X

    return data,d,m,drift_rep,K,T,rep,new_trial,env,algo_list,contextual_features

def get_uniform(d):
    """Get uniform distribution vector of dimension d"""
    return np.full(d, 1 / d)

def find_gt_pf(K,d,X_test_t,y_test_t,models):
    loss_vectors=np.zeros((K,d))
    for i in range(K):
        loss_vectors[i]=test(X_test_t,y_test_t,models[i])
    df=pd.DataFrame(loss_vectors,columns=['acc','auc','f1','ap'])
    pareto_front_df = find_pareto_front(df)  # Find Pareto front
    ground_pf = pareto_front_df.index.to_numpy()
    return ground_pf

def is_dominated(row, other_rows):
    for _, other_row in other_rows.iterrows():
        if all(other_row[1:] <= row[1:]) and any(other_row[1:] < row[1:]):
            return True
    return False

def find_pareto_front(df):
    pareto_front = []
    for index, row in df.iterrows():
        if not is_dominated(row, df.drop(index)):
            pareto_front.append(index)
    return df.loc[pareto_front]

def generate_transition(start_vec, end_vec, num_steps):
    """Generate transition from start_vec to end_vec with num_steps steps"""
    return np.linspace(start_vec, end_vec, num_steps)


def generate_simplex_path(d, n, T):
    """Generate d-dimensional simplex path"""
    # Calculate the time steps for each transition
    num_steps = T // (2 * n * d)

    # Initialize the path with the first one-hot vector
    path = [np.eye(d)[0]]
    for _ in range(T // (2 * num_steps * (d - 1)) + 2):
        for i in range(d - 1):
            # Transition from one-hot to uniform
            path += generate_transition(path[-1], get_uniform(d), num_steps).tolist()[1:]
            # Transition from uniform to next one-hot
            path += generate_transition(path[-1], np.eye(d)[i + 1], num_steps).tolist()[1:]
        # Transition from last one-hot to uniform
        path += generate_transition(path[-1], get_uniform(d), num_steps).tolist()[1:]
        # Transition from uniform to first one-hot
        path += generate_transition(path[-1], np.eye(d)[0], num_steps).tolist()[1:]

    return path


class ParetoFront:
    def __init__(self, ground_pf):
        self.ground_pf = ground_pf
        self.t = 0
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t, X, action_set):
        return random.choice(self.ground_pf)

    def update(self, w_t, X, chosen_arm, mo, inst_regret):
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)

def generate_random_simplex(d):
    # Generate d - 1 random numbers
    random_numbers = np.random.random(d - 1)

    # Add 0 at the start and 1 at the end
    random_numbers = np.concatenate(([0], np.sort(random_numbers), [1]))

    # Take differences of adjacent numbers
    simplex = np.diff(random_numbers)

    return simplex


def generate_weight_sequence(T, d):
    w_t = []
    for i in range(T):
        w_t.append(generate_random_simplex(d))
    w_t = np.array(w_t)
    return w_t


def projection(H, z):
    d = len(z)
    x = cp.Variable(d)
    objective = cp.Minimize(cp.quad_form(x - z, H))
    constraints = [x >= 0, x <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value


def generate_action_set(d, K):
    A = []
    for i in range(K):
        A.append(np.random.rand(d))
    A = np.array(A)
    return A


def generate_feedback(a, mo):
    b = a / mo - a
    feedback = np.random.beta(a, b, 1)[0]
    return feedback


def find_the_best_action(w_t, action_set):
    main_objective = np.dot(action_set, w_t)
    ba = np.argmin(main_objective)
    return ba


def convert_label(y_test):
    y_test = 0.5 * y_test + 0.5
    y_test = np.array(y_test, dtype='int')
    return y_test


def test(X_test, y_test, model):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    avg_prec = average_precision_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    loss_vector = np.array([1 - acc, 1 - f1, 1 - auc, 1 - avg_prec])
    return loss_vector


def fetch_loss_vectors(w_t, K, d, X_test_t, y_test_t, models):
    loss_vectors = np.zeros((K, d))
    for i in range(K):
        loss_vectors[i] = test(X_test_t, y_test_t, models[i])
    objectives = np.dot(w_t, loss_vectors.T)
    return objectives


def find_the_best_model(loss_vectors):
    return np.argmin(loss_vectors)





class EXP3:
    def __init__(self, num_arms, eta):
        self.num_arms = num_arms
        self.eta = eta
        self.weights = np.ones(num_arms)
        self.losses = np.zeros(num_arms)
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t,X,action_set):
        probability_distribution = self.weights / np.sum(self.weights)
        chosen_arm = np.random.choice(self.num_arms, p=probability_distribution)
        #         print (probability_distribution)
        return chosen_arm

    def update(self, w_t, X,chosen_arm, loss, inst_regret):
        estimated_loss = loss / (self.weights[chosen_arm] / np.sum(self.weights))
        self.losses[chosen_arm] += estimated_loss
        self.weights[chosen_arm] = np.exp(-self.eta * self.losses[chosen_arm])
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


class VaRons:
    def __init__(self, K, d, T, gamma):
        self.T = T
        self.K = K
        self.d = d
        self.epsilon = 256 / self.d
        self.L_hat = np.zeros((self.K, self.d))
        self.H = self.epsilon * np.eye(self.d)[None, :] * np.ones((self.K, 1, 1))
        # gamma=1
        self.gamma = gamma
        self.cum_regret, self.cum_regret_hist = 0, []

    def find_lambda(self, pred):
        tol = 1e-3  # tolenrence of error
        gamma = self.gamma
        K = len(pred)
        lb, ub = -1 * min(pred), K / gamma
        err = 1
        while err > tol:
            mid = lb + (ub - lb) / 2
            err = abs(sum(1 / (gamma * (pred + mid))) - 1)
            if sum(1 / (gamma * (pred + mid))) > 1 + tol:
                lb = mid
            elif sum(1 / (gamma * (pred + mid))) < 1 - tol:
                ub = mid
        return mid

    def select_action(self, w_t,X,action_set):
        pred = np.dot(self.L_hat, w_t)
        lambda_t = self.find_lambda(pred)
        p_t = 1 / (self.gamma * (pred + lambda_t))
        p_t = p_t / sum(p_t)
        a_t = random.choices(range(self.K), weights=p_t, k=1)[0]
        return a_t

    def update(self, w_t, X,a_t, y_t, inst_regret):
        err = y_t - np.dot(self.L_hat[a_t], w_t)
        self.H[a_t] += 4 * err ** 2 * np.outer(w_t, w_t)
        pre_proj = self.L_hat[a_t] + 32 * err * np.dot(np.linalg.inv(self.H[a_t]), w_t)
        self.L_hat[a_t] = projection(self.H[a_t], pre_proj)
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


class SWUCB:
    def __init__(self, num_arms, window_size, alpha):
        self.num_arms = num_arms
        self.window_size = window_size
        self.alpha = alpha
        self.rewards = np.zeros((num_arms, window_size))
        self.choices = np.zeros((num_arms, window_size))
        self.t = 0
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t,X,action_set):
        if self.t < self.num_arms:
            return self.t
        else:
            avg_rewards = np.sum(self.rewards, axis=1) / np.sum(self.choices, axis=1)
            confidence_bounds = avg_rewards + np.sqrt(self.alpha * np.log(self.t) / np.sum(self.choices, axis=1))
            return np.argmax(confidence_bounds)

    def update(self, w_t, X,chosen_arm, mo, inst_regret):
        reward = 1 - mo
        if self.t >= self.window_size:
            self.rewards = np.roll(self.rewards, shift=-1, axis=1)
            self.choices = np.roll(self.choices, shift=-1, axis=1)
            self.rewards[:, -1] = 0
            self.choices[:, -1] = 0
        self.rewards[chosen_arm, min(self.t, self.window_size - 1)] = reward
        self.choices[chosen_arm, min(self.t, self.window_size - 1)] = 1
        self.t += 1
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


class DiscountedUCB:
    def __init__(self, num_arms, discount):
        self.num_arms = num_arms
        self.discount = discount
        self.rewards = np.zeros(num_arms)
        self.n_plays = np.zeros(num_arms)
        self.t = 0
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t,X,action_set):
        if self.t < self.num_arms:
            return self.t
        else:
            upper_confidence_bounds = self.rewards / self.n_plays + np.sqrt(2 * np.log(self.t) / self.n_plays)
            return np.argmax(upper_confidence_bounds)

    def update(self, w_t, X,chosen_arm, mo, inst_regret):
        reward = 1 - mo
        self.n_plays[chosen_arm] = self.discount * self.n_plays[chosen_arm] + 1
        self.rewards[chosen_arm] = self.discount * self.rewards[chosen_arm] + reward
        self.t += 1
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


def global_init(K):
    test_data = load_svmlight_file('.\\data\\ijcnn1\\test.dat')
    X_test = test_data[0].toarray()
    y_test = convert_label(np.array(test_data[1]))
    action_set, models = [], []
    model_index = random.sample(range(10), K)
    for i in model_index:
        model_path = './models/model' + str(i) + '.dat'
        cur_model = joblib.load(model_path)
        models.append(cur_model)
        action_set.append(test(X_test, y_test, cur_model))
    action_set = np.array(action_set)
    return X_test, y_test, action_set, models


class ConVaRons:
    def __init__(self, m, d, T, gamma):
        self.T = T
        self.m = m
        self.d = d
        self.beta = min(1 / 4, d / (16 * m))
        self.epsilon = self.d / (self.beta ** 2 * m)
        self.theta_hat = np.zeros(self.d * self.m)
        self.H = self.epsilon * np.eye(self.d * self.m)
        # gamma=1
        self.gamma = gamma
        self.cum_regret, self.cum_regret_hist = 0, []

    def find_lambda(self, pred):
        tol = 1e-3  # tolenrence of error
        gamma = self.gamma
        K = len(pred)
        minimum = min(pred)
        if minimum < 0:
            lb, ub = abs(minimum), K / gamma + max(abs(pred))
        else:
            lb, ub = -1 * minimum, K / gamma + max(abs(pred))
        err = 1
        while err > tol:
            mid = lb + (ub - lb) / 2
            err = abs(sum(1 / (gamma * (pred + mid))) - 1)
            if sum(1 / (gamma * (pred + mid))) > 1 + tol:
                lb = mid
            elif sum(1 / (gamma * (pred + mid))) < 1 - tol:
                ub = mid
        return mid

    def select_action(self, w_t, X, action_set):
        Theta_hat = self.theta_hat.reshape(self.d, self.m)
        pred = np.dot(np.dot(w_t, Theta_hat), X[action_set].T)
        lambda_t = self.find_lambda(pred)
        p_t = 1 / (self.gamma * (pred + lambda_t))
        p_t = p_t / sum(p_t)
        a_t = random.choices(action_set, weights=p_t, k=1)[0]
        return a_t

    def update(self, w_t, X, a_t, y_t, inst_regret):
        z_t = []
        for i in range(self.d):
            for j in range(self.m):
                z_t.append(w_t[i] * X[a_t][j])
        z_t = np.array(z_t)
        err = y_t - np.dot(self.theta_hat, z_t)
        self.H += 4 * err ** 2 * np.outer(z_t, z_t)
        pre_proj = self.theta_hat + 2 / self.beta * err * np.dot(np.linalg.inv(self.H), z_t)
        self.theta_hat = projection(self.H, pre_proj)
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


def find_the_best_action(obj1, obj2, w_t, action_set):
    return min(np.dot(w_t, [obj1[action_set], obj2[action_set]]))


def trial_init(env, algo, K, m,T,d,drift_rep):
    if env == 'random':
        weight_sequence = generate_weight_sequence(T, d)
    elif env == 'drift':
        weight_sequence = generate_simplex_path(d, drift_rep, T)
    if algo == 'EXP3':
        alg = EXP3(K, 20 * np.sqrt(np.log(K) / (K * T)))
    elif algo == 'VaRons':
        alg = VaRons(K, d, T, 20 * K * math.sqrt(T / (d * math.log(T / K))))
    elif algo == 'SW-UCB':
        alg = SWUCB(K, 100, 1)
    elif algo == 'D-UCB':
        alg = DiscountedUCB(K, 0.95)
    elif algo == 'ConVaRons':
        alg = ConVaRons(m, d, T, 20 * K * m * math.sqrt(T / (d * math.log(T))))
    elif algo == 'SWLinUCB':
        alg = SWLinUCB(m, 1, T / math.log(T))
    elif algo == 'WLinUCB':
        alg = WeightedLinUCB(m, 1, 0.97)
    elif algo == 'RestartUCB':
        alg = RestartUCB(m, 200)
    elif algo == 'GT_PF':
        alg = ParetoFront(m)
    return alg, weight_sequence


import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessing(dataset):
    path = "./data/" + dataset + '.csv'
    data = np.genfromtxt(path, delimiter=';')
    df = pd.DataFrame(data)
    if dataset == 'SNW':
        d = 3
    elif dataset == 'NOC':
        df = df.drop([0], axis=0)
        d = 4
    elif dataset == 'LLVM':
        d = 11

    obj1, obj2 = df[:][d] / max(df[:][d]), df[:][d + 1] / max(df[:][d + 1])
    if dataset == 'NOC':
        obj1.index, obj2.index = range(len(obj1)), range(len(obj1))
    X = df.drop([d, d + 1], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.c_[X, np.ones(X.shape[0])]
    return X, obj1, obj2


class ConVaRons:
    def __init__(self, m, d, T, gamma):
        self.T = T
        self.m = m
        self.d = d
        self.beta = min(1 / 4, d / (16 * m))
        self.epsilon = self.d / (self.beta ** 2 * m)
        self.theta_hat = np.zeros(self.d * self.m)
        self.H = self.epsilon * np.eye(self.d * self.m)
        # gamma=1
        self.gamma = gamma
        self.cum_regret, self.cum_regret_hist = 0, []

    def find_lambda(self, pred):
        tol = 1e-3  # tolenrence of error
        gamma = self.gamma
        K = len(pred)
        minimum = min(pred)
        if minimum < 0:
            lb, ub = abs(minimum), K / gamma + max(abs(pred))
        else:
            lb, ub = -1 * minimum, K / gamma + max(abs(pred))
        err = 1
        while err > tol:
            mid = lb + (ub - lb) / 2
            err = abs(sum(1 / (gamma * (pred + mid))) - 1)
            if sum(1 / (gamma * (pred + mid))) > 1 + tol:
                lb = mid
            elif sum(1 / (gamma * (pred + mid))) < 1 - tol:
                ub = mid
        return mid

    def select_action(self, w_t, X, action_set):
        Theta_hat = self.theta_hat.reshape(self.d, self.m)
        pred = np.dot(np.dot(w_t, Theta_hat), X[action_set].T)
        lambda_t = self.find_lambda(pred)
        p_t = 1 / (self.gamma * (pred + lambda_t))
        p_t = p_t / sum(p_t)
        a_t = random.choices(action_set, weights=p_t, k=1)[0]
        return a_t

    def update(self, w_t, X, a_t, y_t, inst_regret):
        z_t = []
        for i in range(self.d):
            for j in range(self.m):
                z_t.append(w_t[i] * X[a_t][j])
        z_t = np.array(z_t)
        err = y_t - np.dot(self.theta_hat, z_t)
        self.H += 4 * err ** 2 * np.outer(z_t, z_t)
        pre_proj = self.theta_hat + 2 / self.beta * err * np.dot(np.linalg.inv(self.H), z_t)
        self.theta_hat = projection(self.H, pre_proj)
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


import numpy as np


class SWLinUCB:
    def __init__(self, m, alpha, window_size):
        self.m = m  # dimension of the context
        self.alpha = alpha  # parameter controlling the exploration
        self.window_size = window_size  # size of the sliding window
        self.A = np.eye(m)  # A matrix
        self.b = np.zeros(m)  # b vector
        self.A_inv = np.eye(m)  # Inverse of A
        self.recent_rewards = []
        self.recent_actions = []
        self.recent_contexts = []
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t, X, action_set):
        # Compute upper confidence bounds for all arms
        ucb_values = []
        for a in action_set:
            theta = np.dot(self.A_inv, self.b)
            p = np.dot(theta, X[a]) + self.alpha * np.sqrt(np.dot(np.dot(X[a], self.A_inv), X[a]))
            ucb_values.append(p)
        return action_set[np.argmax(ucb_values)]

    def update(self, w_t, X, a_t, y_t, inst_regret):
        r_t = 1 - y_t
        # Update only if the chosen action is within the window size
        if len(self.recent_actions) >= self.window_size:
            old_a = self.recent_actions.pop(0)
            old_r = self.recent_rewards.pop(0)
            old_x = self.recent_contexts.pop(0)
            # Undo old updates
            self.A -= np.outer(X[old_a], X[old_a])
            self.b -= old_r * X[old_a]

        # Add new context, reward and action to recent memory
        self.recent_contexts.append(X[a_t])
        self.recent_rewards.append(r_t)
        self.recent_actions.append(a_t)

        # Update A and b
        self.A += np.outer(X[a_t], X[a_t])
        self.b += r_t * X[a_t]
        self.A_inv = np.linalg.inv(self.A)
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)




class WeightedLinUCB:
    def __init__(self, m, alpha, discount_factor):
        self.m = m  # dimension of the context
        self.alpha = alpha  # parameter controlling the exploration
        self.discount_factor = discount_factor  # discount factor
        self.A = np.eye(m)  # A matrix
        self.b = np.zeros(m)  # b vector
        self.t = 0  # time step
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t, X, action_set):
        # Compute inverse of A
        A_inv = np.linalg.inv(self.A)

        # Compute upper confidence bounds for all arms
        ucb_values = []
        for a in action_set:
            theta = np.dot(A_inv, self.b)
            p = np.dot(X[a], theta) + self.alpha * np.sqrt(np.dot(np.dot(X[a], A_inv), X[a]))
            ucb_values.append(p)
        return action_set[np.argmax(ucb_values)]

    def update(self, w_t, X, a_t, y_t, inst_regret):
        r_t = 1 - y_t
        # Calculate weight
        w = self.discount_factor ** self.t

        # Update A and b
        self.A += w * np.outer(X[a_t], X[a_t])
        self.b += w * r_t * X[a_t]

        # Increment time step
        self.t += 1
        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)


class RestartUCB:
    def __init__(self, m, H):
        self.m = m  # dimension of the context
        self.H = H  # restart period
        self.alpha = 1  # parameter controlling the exploration
        self.discount_factor = 1  # discount factor
        self.A = np.eye(m)  # A matrix
        self.b = np.zeros(m)  # b vector
        self.t = 0  # time step
        self.cum_regret, self.cum_regret_hist = 0, []

    def select_action(self, w_t, X, action_set):
        # Compute inverse of A
        A_inv = np.linalg.inv(self.A)

        # Compute upper confidence bounds for all arms
        ucb_values = []
        for a in action_set:
            theta = np.dot(A_inv, self.b)
            p = np.dot(X[a], theta) + self.alpha * np.sqrt(np.dot(np.dot(X[a], A_inv), X[a]))
            ucb_values.append(p)
        return action_set[np.argmax(ucb_values)]

    def update(self, w_t, X, a_t, y_t, inst_regret):
        r_t = 1 - y_t
        # Calculate weight
        w = self.discount_factor ** self.t

        # Update A and b
        self.A += w * np.outer(X[a_t], X[a_t])
        self.b += w * r_t * X[a_t]

        # Increment time step
        self.t += 1
        if self.t == self.H:
            self.A = np.eye(self.m)  # A matrix
            self.b = np.zeros(self.m)  # b vector
            self.t = 0  # restart

        self.cum_regret += inst_regret
        self.cum_regret_hist.append(self.cum_regret)

    def fetch_regret(self):
        return self.cum_regret, np.array(self.cum_regret_hist)

