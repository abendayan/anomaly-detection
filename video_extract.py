import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join

N = 5

PATH = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001/'
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
files.remove('.DS_Store')
files.remove('._.DS_Store')
files.sort()
fgbg = cv2.createBackgroundSubtractorMOG2()
all_ceils = []
number_frame = 0
for one_file in files:
    print number_frame
    frame = cv2.imread(PATH + one_file)
    fgmask = fgbg.apply(frame)
    width = fgmask.shape[0]
    height = fgmask.shape[1]
    #for i in fgmask:
    #    if np.any(fgmasik)
    ceils = [[0 for x in range(height/N)] for y in range(width/N)]
    # print width/N
    for i in range(width/N):
        for j in range(height/N):
            if not np.all(fgmask[i:i+N,j:j+N] == 127):
                ceils[i][j] = fgmask[i:i+N,j:j+N]
                # import pdb; pdb.set_trace()
    all_ceils.append(ceils)
    # import pdb; pdb.set_trace()
    cv2.imshow('frame', fgmask)
    number_frame += 1
    # cv2.waitKey(0)
    if number_frame == len(files):
        import pdb; pdb.set_trace()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
