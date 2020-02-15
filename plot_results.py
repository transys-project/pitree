import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


RESULTS_FOLDER = './results/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 80
COLOR_MAP = plt.cm.jet  # nipy_spectral, Set1,Paired
SCHEMES = []


def main(args):
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    if args.pensieve:
        SCHEMES.append('pensieve')
    if args.pensiedt:
        SCHEMES.append('pensiedt')
    if args.robustmpc:
        SCHEMES.append('robustmpc')
    if args.robustmdt:
        SCHEMES.append('robustmdt')
    if args.hotdash:
        SCHEMES.append('hotdash')
    if args.hotdadt:
        SCHEMES.append('hotdadt')

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        # print(log_file)

        with open(RESULTS_FOLDER + log_file, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                parse = line.split('\t')
                if len(parse) <= 1:
                    continue
                # break
                # [time_ms, bit_rate, buff, volume, time, reward]
                time_ms.append(float(parse[0]))
                bit_rate.append(float(parse[1]))
                buff.append(float(parse[2]))
                bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                reward.append(float(parse[6]))

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []

    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < 10:
                schemes_check = False
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:min(VIDEO_LEN, len(time_all[scheme][l]))]))

    mean_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])

    print(mean_rewards)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for scheme in SCHEMES:
    #     ax.plot(reward_all[scheme])
    #
    # SCHEMES_REW = []
    # for scheme in SCHEMES:
    #     SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))
    #
    # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # for i,j in enumerate(ax.lines):
    #     j.set_color(colors[i])
    #
    # ax.legend(SCHEMES_REW, loc=4)
    #
    # plt.ylabel('total reward')
    # plt.xlabel('trace index')
    # plt.show()

    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # for scheme in SCHEMES:
    #     values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
    #     cumulative = np.cumsum(values)
    #     ax.plot(base[:-1], cumulative)

    # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # for i,j in enumerate(ax.lines):
    #     j.set_color(colors[i])

    # ax.legend(SCHEMES, loc=4)

    # plt.ylabel('CDF')
    # plt.xlabel('total reward')
    # plt.show()

    # ---- ---- ---- ----
    # check each trace
    # ---- ---- ---- ----

    # for l in time_all[SCHEMES[0]]:
    # 	schemes_check = True
    # 	for scheme in SCHEMES:
    # 		if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
    # 			schemes_check = False
    # 			break
    # 	if schemes_check:
    # 		fig = plt.figure()
    #
    # 		ax = fig.add_subplot(311)
    # 		for scheme in SCHEMES:
    # 			ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
    # 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # 		for i,j in enumerate(ax.lines):
    # 			j.set_color(colors[i])
    # 		plt.title(l)
    # 		plt.ylabel('bit rate selection (kbps)')
    #
    # 		ax = fig.add_subplot(312)
    # 		for scheme in SCHEMES:
    # 			ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
    # 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # 		for i,j in enumerate(ax.lines):
    # 			j.set_color(colors[i])
    # 		plt.ylabel('buffer size (sec)')
    #
    # 		ax = fig.add_subplot(313)
    # 		for scheme in SCHEMES:
    # 			ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
    # 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # 		for i,j in enumerate(ax.lines):
    # 			j.set_color(colors[i])
    # 		plt.ylabel('bandwidth (mbps)')
    # 		plt.xlabel('time (sec)')
    #
    # 		SCHEMES_REW = []
    # 		for scheme in SCHEMES:
    # 			SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))
    #
    # 		ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
    # 		plt.show()

    for scheme in SCHEMES:
        reward_all[scheme].sort()

    # ---- ---- ---- ----
    # save sorted reward
    # ---- ---- ---- ----

    for scheme in SCHEMES:
        with open('rewards/' + scheme + '.csv', 'w') as f:
            for reward in reward_all[scheme]:
                f.write(str(reward) + '\n')

    # ---- ---- ---- ----
    # save faithfulness
    # ---- ---- ---- ----

    # if args.robustmpc and args.robustmdt:
    #     with open('faithfulness/robustmpc.csv', 'w') as f:
    #         for index in range(len(reward_all['robustmpc'])):
    #             f.write(str((reward_all['robustmdt'][index] - reward_all['robustmpc'][index]) /
    #                         np.abs(mean_rewards['robustmpc']) + 1) + '\n')
    # if args.pensieve and args.pensiedt:
    #     with open('faithfulness/pensieve.csv', 'w') as f:
    #         for index in range(len(reward_all['pensieve'])):
    #             f.write(str((reward_all['pensiedt'][index] - reward_all['pensieve'][index]) /
    #                         np.abs(mean_rewards['pensieve']) + 1) + '\n')
    # if args.hotdash and args.hotdadt:
    #     with open('faithfulness/hotdash.csv', 'w') as f:
    #         for index in range(len(reward_all['hotdash'])):
    #             f.write(str((reward_all['hotdadt'][index] - reward_all['hotdash'][index]) /
    #                         np.abs(mean_rewards['hotdash']) + 1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', '--pensieve', action='store_true')
    parser.add_argument('-pd', '--pensiedt', action='store_true')
    parser.add_argument('-rn', '--robustmpc', action='store_true')
    parser.add_argument('-rd', '--robustmdt', action='store_true')
    parser.add_argument('-hn', '--hotdash', action='store_true')
    parser.add_argument('-hd', '--hotdadt', action='store_true')
    args = parser.parse_args()
    main(args)
