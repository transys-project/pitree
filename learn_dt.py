import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import pickle as pk
import csv
import pensieve
import pensiedt
import robustmpc
import robustmdt
import hotdash
import hotdadt
import argparse
import load_trace
import fixed_env as env
import fixed_env_hotdash as env_hotdash
from multiprocessing import Pool
import time


S_INFO_P = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO_R = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_ABR_INFO_H = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_HOT_INFO_H = 6  # next_hs_chunk_size, num_hs_remaining, num_chunks_remaining_till_hs_chunk_played, play_buffer_size, bitrate_last_hs, dist_vector_from_hs_chunks
S_BRT_INFO_H = 2  # next_bit_rate, next_hs_bit_rate
S_INFO_H = S_ABR_INFO_H + S_HOT_INFO_H + S_BRT_INFO_H
S_LEN = 8  # take how many frames in the past
A_DIM_P = 6
A_DIM_H = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
BITRATE_LEVELS = 6
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
DEFAULT_PREFETCH = 0 # default prefetch decision without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_pensieve'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/pretrain_linear_reward.ckpt'


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


def get_rollouts(env, policy, args, n_batch_rollouts, dt_policy=None):
    rollouts = []
    if dt_policy is None:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env))
    else:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env, dt_policy))
    return rollouts


def resample(states, actions, serials, max_pts):
    idx = np.random.choice(len(states), size=max_pts)
    return states[idx], actions[idx], serials[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-n', '--leaf-nodes', type=int, default=100)
    parser.add_argument('-q', '--qoe-metric', choices=['lin', 'log', 'hd'], default='lin')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-d', '--dt', action='store_true')
    parser.add_argument('-w', '--worker', type=int, default=1)
    parser.add_argument('-t', '--traces', choices=['norway', 'fcc', 'oboe'])
    parser.add_argument('-i', '--iters', type=int, default=500)
    parser.add_argument('-v', '--visualize', type=bool, default=False)

    args = parser.parse_args()
    n_batch_rollouts = 10
    max_iters = args.iters
    pts = 200000
    train_frac = 0.8
    np.random.seed(RANDOM_SEED)
    states, actions, serials = [], [], []
    trees = []
    precision = []
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
    if args.abr == 'hotdash':
        net_env = env_hotdash.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names)
    else:
        net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names)

    time_calc = np.zeros((max_iters, 3))

    if args.abr == 'pensieve':
        teacher = pensieve.Pensieve()
        student = pensiedt.PensieveDT()
        predict = teacher.predict
    elif args.abr == 'robustmpc':
        teacher = robustmpc.RobustMPC()
        student = robustmdt.RobustMPCDT()
        predict = teacher.predict
    elif args.abr == 'hotdash':
        teacher = hotdash.Hotdash()
        student = hotdadt.HotdashDT()
        predict = teacher.predict
    else:
        raise NotImplementedError

    t1 = time.time()

    # Step 1: Initialization for the first iteration
    trace = get_rollouts(env=net_env, policy=teacher, args=args, n_batch_rollouts=n_batch_rollouts)
    states.extend((state for state, _, _ in trace))
    actions.extend((action for _, action, _ in trace))
    serials.extend(serial for _, _, serial in trace)

    for i in range(max_iters):
        # Step 2:
        print('Iteration {}/{}'.format(i, max_iters))

        # Step 2a: Resample or not.
        cur_states, cur_actions, cur_serials = resample(np.array(states), np.array(actions), np.array(serials), pts)
        serials_train, actions_train, serials_val, actions_val = split_train_test(cur_serials, cur_actions, train_frac)
        dt_policy = DecisionTreeClassifier(max_leaf_nodes=args.leaf_nodes)
        dt_policy.fit(serials_train, actions_train)

        t4 = time.time()
        precision.append(np.mean(dt_policy.predict(serials_val) == actions_val))
        print('unpruned precision', precision[-1])
        t5 = time.time()

        reward = 0

        t2 = time.time()
        time_calc[i][0] = t2 - t1 + t4 - t5

        student_trace = get_rollouts(env=net_env, policy=student, args=args, n_batch_rollouts=n_batch_rollouts,
                                     dt_policy=dt_policy)
        student_states = [state for state, _, _ in student_trace]
        student_actions = [action for _, action, _ in student_trace]
        student_serials = [serial for _, _, serial in student_trace]

        t3 = time.time()
        time_calc[i][1] = t3 - t2

        if args.abr == 'pensieve' or args.abr == 'hotdash':
            teacher_actions = map(predict, student_states)
        else:
            pool = Pool(args.worker)
            teacher_actions = pool.map(predict, student_states)
            pool.close()
            pool.join()

        states.extend(student_states)
        actions.extend(teacher_actions)
        serials.extend(student_serials)

        t1 = time.time()
        time_calc[i][2] = t1 - t3

        trees.append((dt_policy, reward))

    best_tree, max_reward = trees[-1]

    # You can further optimize the decision tree by finding the optimal tree among all iterations. 
    # However, experiences show that the final one is always the best one.

    # best_tree = None
    # max_reward = 0
    # for (dt_policy, reward) in trees:
    #     if reward > max_reward:
    #         best_tree = dt_policy
    #         max_reward = reward

    # save decision tree to file
    with open('decision_tree/' + args.abr + '_' + args.traces + '_' + str(args.leaf_nodes) + '.pk3', 'wb') as f:
        pk.dump(best_tree, f)

    if args.visualize:
        dot_data = StringIO()
        export_graphviz(best_tree, out_file=dot_data, filled=True)
        out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        out_graph.write_svg('tree/' + args.abr + '.svg')

    with open('precision/' + args.abr + '_' + args.traces + '_' + str(args.leaf_nodes) + '.csv', 'wb') as f:
        for i in precision:
            f.write(bytes(str(i) + '\n', encoding='utf-8'))

    with open('time/' + args.abr + '_' + args.traces + '_' + str(args.leaf_nodes) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for time_breakdown in time_calc:
            writer.writerow(time_breakdown)
