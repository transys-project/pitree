import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.process
import tornado.netutil
from concurrent.futures import ThreadPoolExecutor
import json
import argparse
import numpy as np
import time
import socket

import robustmpc
import pensieve
import hotdash


IP_PORT = 9999

S_INFO_R = 5
S_LEN = 8
S_ABR_INFO = 6
S_HOT_INFO = 6
S_BRT_INFO = 2
S_INFO_H = S_ABR_INFO + S_HOT_INFO + S_BRT_INFO
S_INFO_PENSIEVE = 6
A_DIM = 6
A_DIM_prefetch = 2
S_INFO_bitr = 6
S_INFO_P = 6

MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
ENTROPY_CHANGE_INTERVAL = 20000
HD_REWARD = [1, 2, 3, 12, 15, 20]
NUM_HOTSPOT_CHUNKS = 5
BITRATE_LEVELS = 6
DEFAULT_PREFETCH = 0 # default prefetch decision without agent
RAND_RANGE = 1000


class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(20)

    def initialize(self, args, teacher):
        self.teacher = teacher
        self.args = args

    def post(self):
        t1 = time.time()
        env_post_data = json.loads(self.request.body)
        last_bit_rate = env_post_data['last_bit_rate']
        buffer_size = env_post_data['buffer_size']
        rebuf = env_post_data['rebuf']
        video_chunk_size = env_post_data['video_chunk_size']
        delay = env_post_data['delay']
        video_chunk_remain = env_post_data['video_chunk_remain']
        next_video_chunk_sizes = env_post_data['next_video_chunk_sizes']

        if self.args.abr == 'pensieve':
            state = np.zeros((S_INFO_P, S_LEN))
            state[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            bit_rate = int(self.teacher.predict(state))

        elif self.args.abr == 'robustmpc':
            state = np.zeros((S_INFO_R, S_LEN))
            state[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            bit_rate = int(self.teacher.predict(state))

        elif self.args.abr == 'hotdash':
            hotspot_chunks_remain = env_post_data['hotspot_chunks_remain']
            last_hotspot_bit_rate = env_post_data['last_hotspot_bit_rate']
            next_hotspot_chunk_sizes = env_post_data['next_hotspot_chunk_sizes']
            dist_from_hotspot_chunks = env_post_data['dist_from_hotspot_chunks']

            state = np.zeros((S_INFO_H, S_LEN))
            state[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            state[6, -1] = np.minimum(hotspot_chunks_remain, NUM_HOTSPOT_CHUNKS) / float(NUM_HOTSPOT_CHUNKS)
            state[7, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            state[8, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[9, -1] = last_hotspot_bit_rate / float(np.max(VIDEO_BIT_RATE))
            state[10, :BITRATE_LEVELS] = np.array(next_hotspot_chunk_sizes) / M_IN_K / M_IN_K
            state[11, :NUM_HOTSPOT_CHUNKS] = (np.array(
                dist_from_hotspot_chunks) + CHUNK_TIL_VIDEO_END_CAP) / 2 / CHUNK_TIL_VIDEO_END_CAP
            state[12, -1] = last_bit_rate / float(np.max(VIDEO_BIT_RATE))
            state[13, -1] = last_hotspot_bit_rate / float(np.max(VIDEO_BIT_RATE))

            state_info_pensieve_n = np.zeros((S_INFO_PENSIEVE, S_LEN))
            state_info_pensieve_n[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))
            state_info_pensieve_n[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state_info_pensieve_n[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state_info_pensieve_n[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
            state_info_pensieve_n[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            state_info_pensieve_n[5, -1] = np.minimum(video_chunk_remain,
                                                      CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            state_info_pensieve_h = np.zeros((S_INFO_PENSIEVE, S_LEN))
            state_info_pensieve_h[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))
            state_info_pensieve_h[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_h[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state_info_pensieve_h[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_h[4, :BITRATE_LEVELS] = np.array(next_hotspot_chunk_sizes) / M_IN_K / M_IN_K
            state_info_pensieve_h[5, -1] = np.minimum(video_chunk_remain,
                                                      CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            states_list = np.array([state, state_info_pensieve_n, state_info_pensieve_h])
            bit_rate = int(self.teacher.predict(states_list)[1])

        else:
            raise NotImplementedError

        send_data = json.dumps({"bitrate": bit_rate})
        self.set_status(200)
        self.set_header('Content-Type', 'text/plain')
        self.set_header('Content-Length', len(send_data))
        self.set_header('Access-Control-Allow-Origin', "*")
        self.write(bytes(send_data, encoding='utf-8'))
        t2 = time.time()
        print(t2 - t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-w', '--worker', type=int, default=1)
    args = parser.parse_args()

    sockets = tornado.netutil.bind_sockets(IP_PORT)
    tornado.process.fork_processes(args.worker)
    if args.abr == 'pensieve':
        teacher = pensieve.Pensieve()
    elif args.abr == 'robustmpc':
        teacher = robustmpc.RobustMPC()
    elif args.abr == 'hotdash':
        teacher = hotdash.Hotdash()
    else:
        raise NotImplementedError
    application = tornado.web.Application(handlers=[(r"/", MainHandler, dict(args=args, teacher=teacher))],
                                          autoreload=False, debug=False)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.add_sockets(sockets)

    tornado.ioloop.IOLoop.current().start()
