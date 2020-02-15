import numpy as np
import fixed_env as env
import load_trace
import matplotlib.pyplot as plt
import itertools
from get_reward import get_reward
from get_chunk_size import get_chunk_size

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5

VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 80.0
TOTAL_VIDEO_CHUNKS = 80
M_IN_K = 1000.0
REBUF_PENALTY = {'lin': 40, 'log': 3.69, 'hd': 8}  # 1 sec rebuffering -> 40 Mbps
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
LOG_FILE = './results/log_robustmpc'

CHUNK_COMBO_OPTIONS = []
for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
    CHUNK_COMBO_OPTIONS.append(combo)

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []


class RobustMPC:
    def __init__(self):
        pass

    def main(self, args, net_env=None):
        self.args = args
        np.random.seed(RANDOM_SEED)
        viper_flag = True
        assert len(VIDEO_BIT_RATE) == A_DIM

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

            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, \
            video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = get_reward(bit_rate, rebuf, last_bit_rate, args.qoe_metric)
            r_batch.append(reward)
            last_bit_rate = bit_rate

            if args.log:
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
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            bit_rate = self.predict(state)
            serialized_state = []
            # Log input of neural network
            serialized_state.append(state[0, -1])
            serialized_state.append(state[1, -1])
            serialized_state.append(state[2, -1])
            for i in range(5):
                serialized_state.append(state[3, i])
            serialized_state.append(state[4, -1])

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

    def predict(self, state):
        qoe_metric = 'lin'
        buffer_size = state[1, -1] * BUFFER_NORM_FACTOR
        bit_rate = np.where(state[0, -1] * np.max(VIDEO_BIT_RATE) == VIDEO_BIT_RATE)[0][0]
        video_chunk_remain = state[4, -1] * CHUNK_TIL_VIDEO_END_CAP
        # ================== MPC =========================
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if len(past_bandwidth_ests) > 0:
            curr_error = abs(past_bandwidth_ests[-1] - state[3, -1]) / float(state[3, -1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if len(past_errors) < 5:
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)

        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if TOTAL_VIDEO_CHUNKS - last_index < 5:
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size

        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int(bit_rate)
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (get_chunk_size(chunk_quality, index)
                                 / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                if curr_buffer < download_time:
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                if qoe_metric == 'lin':
                    bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                elif qoe_metric == 'log':
                    bitrate_sum += np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                    smoothness_diffs += abs(np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[last_quality])))
                elif qoe_metric == 'hd':
                    bitrate_sum += BITRATE_REWARD[chunk_quality]
                    smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            reward = (bitrate_sum / 1000.) - (REBUF_PENALTY[qoe_metric] * curr_rebuffer_time) - (smoothness_diffs / 1000.)

            if reward >= max_reward:
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if best_combo != ():  # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        return bit_rate
