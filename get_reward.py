import numpy as np


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000


def get_reward(bit_rate, rebuf, last_bit_rate, reward_type):
    if reward_type == 'lin':
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - \
                 SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
    elif reward_type == 'log':
        log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
        log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))
        reward = log_bit_rate - REBUF_PENALTY * rebuf - REBUF_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
    elif reward_type == 'hd':
        reward = HD_REWARD[bit_rate] - REBUF_PENALTY * rebuf - \
                 SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])
    else:
        raise NotImplementedError
    return reward
