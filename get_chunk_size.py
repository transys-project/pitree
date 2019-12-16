import numpy as np


size_video = [np.loadtxt('video/video_size_' + str(i), dtype=int).tolist() for i in range(6)]


def get_chunk_size(quality, index):
    if index < 0 or index > 48:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    return size_video[quality][index]
