from pydub import AudioSegment
from multiprocessing import Pool
import os
import sys
import glob

def convert(pathname, size = 10):
    try:
        song = AudioSegment.from_mp3(pathname)
        split_path = os.path.split(pathname)
        os.mkdir(os.path.join(split_path[0], split_path[1][:-4]))
        for i, start in enumerate(range(1, len(song), size*1000)):
            cur_segment = song[start: start + size*1000]
            cur_segment.export(os.path.join(split_path[0], split_path[1][:-4], "segment_{}.wav".format(i)), format = "wav")

        return 0
    except Execption as e:
        with open("convert.log", "w+") as log:
            log.write(pathname)
            log.write(e)
            

if __name__ == "__main__":
    for path in glob.glob("./**/*.mp3", recursive = True):
        convert(path)
 
