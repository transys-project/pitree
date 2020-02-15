import numpy as np


VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0
REBUF_PENALTY = {'lin': 40, 'log': 3.69, 'hd': 8}  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent


def get_reward(bit_rate, rebuf, last_bit_rate, reward_type):
    if reward_type == 'lin':
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY[reward_type] * rebuf - \
                 SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
    elif reward_type == 'log':
        log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
        log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))
        reward = log_bit_rate - REBUF_PENALTY[reward_type] * rebuf - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
    elif reward_type == 'hd':
        reward = HD_REWARD[bit_rate] - REBUF_PENALTY[reward_type] * rebuf - \
                 SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])
    else:
        raise NotImplementedError
    return reward
