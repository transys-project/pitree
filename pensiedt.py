import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle as pk
import fixed_env as env
import load_trace
from get_reward import get_reward
import fsm


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

CHUNK_ON = 0
CHUNK_SWITCH = 1
OPTIMIZED = 2
HORIZON = 5
CHUNK_LEN = 4.0
BITRATE_NUM = 6

LOG_FILE = 'results/log_pensiedt'


class PensieveDT:
    def __init__(self):
        pass

    def main(self, args, net_env=None, policy=None):
        viper_flag = True
        assert len(VIDEO_BIT_RATE) == A_DIM
        log_f = LOG_FILE

        if net_env is None:
            viper_flag = False
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
            net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                                      all_file_names=all_file_names)
        if args.update:
            log_f = log_f.replace('dt', 'du')

        if not viper_flag and args.log:
            log_path = log_f + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
            log_file = open(log_path, 'wb')

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        rollout = []
        video_count = 0
        reward_sum = 0
        in_compute = []

        # load dt policy
        if policy is None:
            with open(args.dt, 'rb') as f:
                policy = pk.load(f)
        policy = fsm.FSM(policy)

        while True:  # serve video forever
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, \
            video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = get_reward(bit_rate, rebuf, last_bit_rate, args.qoe_metric)
            r_batch.append(reward)
            reward_sum += reward
            last_bit_rate = bit_rate

            if args.log:
                log_file.write(bytes(str(time_stamp / M_IN_K) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n', encoding='utf-8'))
                log_file.flush()


            # select bit_rate according to decision tree
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            serialized_state = serial(state)
            bit_rate = int(policy.predict(np.array(serialized_state).reshape(1, -1))[0])
            rollout.append((state, bit_rate, serialized_state))
            s_batch.append(state)

            if args.update:
                chunk_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
                policy.chunk_leaf[chunk_index] = policy.tree.apply(np.array(serialized_state).reshape(1, -1))
                if chunk_index < CHUNK_TIL_VIDEO_END_CAP - HORIZON:
                    in_compute.append(fsm.Trajectory(chunk_index, max(0, bit_rate - 1), buffer_size - CHUNK_LEN,
                                                     last_bit_rate, state, args))
                    in_compute.append(fsm.Trajectory(chunk_index, bit_rate, buffer_size - CHUNK_LEN,
                                                     last_bit_rate, state, args))
                    in_compute.append(fsm.Trajectory(chunk_index, min(5, bit_rate + 1), buffer_size - CHUNK_LEN,
                                                     last_bit_rate, state, args))

                for traj in in_compute:
                    this_chunk_size = video_chunk_size
                    this_delay = delay
                    while True:
                        if traj.apply(this_chunk_size, this_delay) == CHUNK_SWITCH:
                            new_bitrate = int(policy.predict(np.array(serial(traj.states)).reshape(1, -1))[0])
                            traj.next_chunk(new_bitrate)
                            this_chunk_size, this_delay = traj.trans_msg
                        else:
                            break

                    while len(in_compute) > 1 and in_compute[0].end and in_compute[1].end and in_compute[2].end:
                        r_below = sum([get_reward(in_compute[0].quality[i], in_compute[0].rebuf[i],
                                                  in_compute[0].last_bitrate[i], args.qoe_metric) for i in range(HORIZON)])
                        r_normal = sum([get_reward(in_compute[1].quality[i], in_compute[1].rebuf[i],
                                                  in_compute[1].last_bitrate[i], args.qoe_metric) for i in range(HORIZON)])
                        r_above = sum([get_reward(in_compute[2].quality[i], in_compute[2].rebuf[i],
                                                  in_compute[2].last_bitrate[i], args.qoe_metric) for i in range(HORIZON)])
                        if r_above == max(r_below, r_normal, r_above):
                            policy.update(in_compute[0].chunk_index, 1)
                        elif r_normal == max(r_below, r_normal, r_above):
                            policy.update(in_compute[0].chunk_index, -1)
                        else:
                            policy.update(in_compute[0].chunk_index, 0)

                        in_compute.pop(0)
                        in_compute.pop(0)
                        in_compute.pop(0)

            if end_of_video:
                if args.log:
                    log_file.write(bytes('\n', encoding='utf-8'))
                    log_file.close()
                    print("video count", video_count)

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                r_batch = []
                in_compute = []

                if viper_flag:
                    return rollout
                else:
                    video_count += 1
                    if video_count >= len(net_env.all_file_names):
                        break
                    if args.log:
                        log_path = log_f + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
                        log_file = open(log_path, 'wb')

        return reward_sum


def serial(state):
    serialized_state = []
    serialized_state.append(state[0, -1])
    serialized_state.append(state[1, -1])
    for i in range(S_LEN):
        serialized_state.append(state[2, i])
    for i in range(S_LEN):
        serialized_state.append(state[3, i])
    for i in range(A_DIM):
        serialized_state.append(state[4, i])
    serialized_state.append(state[5, -1])
    return serialized_state
