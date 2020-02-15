import os
import numpy as np
import tensorflow as tf
import fixed_env as env
import a3c
import load_trace
from get_reward import get_reward

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 80.0
M_IN_K = 1000.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_pensieve'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/pensieve.ckpt'


class Pensieve:
    def __init__(self):
        self.sess = tf.Session()
        self.actor = a3c.ActorNetwork(self.sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                      learning_rate=ACTOR_LR_RATE)
        self.critic = a3c.CriticNetwork(self.sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)

        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, NN_MODEL)

    def main(self, args, net_env=None):

        np.random.seed(RANDOM_SEED)
        viper_flag = True
        assert len(VIDEO_BIT_RATE) == A_DIM

        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)

        if net_env is None:
            viper_flag = False
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
            net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                                      all_file_names=all_file_names)

        if not viper_flag and args.log:
            log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
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
        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = get_reward(bit_rate, rebuf, last_bit_rate, args.qoe_metric)
            r_batch.append(reward)
            last_bit_rate = bit_rate

            if args.log:
                # [time_ms, bit_rate, buff, volume, time, reward]
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(bytes(str(time_stamp / M_IN_K) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n', encoding='utf-8'))
                log_file.flush()

            # retrieve previous state
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

            bit_rate = self.predict(state)
            s_batch.append(state)

            serialized_state = []
            # Log input of neural network
            serialized_state.append(state[0, -1])
            serialized_state.append(state[1, -1])
            for i in range(S_LEN):
                serialized_state.append(state[2, i])
            for i in range(S_LEN):
                serialized_state.append(state[3, i])
            for i in range(A_DIM):
                serialized_state.append(state[4, i])
            serialized_state.append(state[5, -1])

            rollout.append((state, bit_rate, serialized_state))
            if end_of_video:
                if args.log:
                    log_file.write(bytes('\n', encoding='utf-8'))
                    log_file.close()
                    print("video count", video_count)

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                if viper_flag:
                    break
                else:
                    video_count += 1
                    if video_count >= len(net_env.all_file_names):
                        break
                    if args.log:
                        log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
                        log_file = open(log_path, 'wb')

        return rollout

    def predict(self, states):
        action_prob = self.actor.predict(np.reshape(states, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        return bit_rate
