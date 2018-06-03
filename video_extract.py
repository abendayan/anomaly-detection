import sys
import numpy as np
import time
import cv2
import pmf as pmf
from os import listdir
from os.path import isfile, join
from scipy.stats.kde import gaussian_kde

# https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py

def remove_duplicates(l):
    return list(set(l))

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

DETECT_INTERVAL = 10
N = 16
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

PATH = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001/'
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
files.remove('.DS_Store')
files.remove('._.DS_Store')
files.sort()
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
all_ceils = []
old_frame = None
tracks = []
movement = 0
number_frame = 0

mots = []
tracks_ceils_id = []

pmfSpeed = pmf.Pmf()
avgMots = []
for one_file in files:
    frame = cv2.imread(PATH + one_file)
    fgmask = fgbg.apply(frame)
    width = fgmask.shape[0]
    height = fgmask.shape[1]
    vis = frame.copy()
    ceils = [[None for x in range(height/N + 1)] for y in range(width/N + 1)]
    mot = [[0 for x in range(height/N + 1)] for y in range(width/N + 1)]
    for i in range(0, width, N):
        for j in range(0, height, N):
            if np.any(fgmask[i:i+N,j:j+N] == 255):
                ceils[i/N][j/N] = fgmask[i:i+N,j:j+N]
                if number_frame % DETECT_INTERVAL == 0:
                    # print len(tracks)
                    # if number_frame == 2*DETECT_INTERVAL:
                        # import pdb; pdb.set_trace()
                    to_add= True
                    for tr in tracks:
                        if int(tr[-1][0]) <= i+N and int(tr[-1][0]) >= i and int(tr[-1][1]) <= j+N  and int(tr[-1][1]) >= j:
                            to_add = False
                            # print "a"

                    if to_add and not [(i+N/2, j+N/2)] in tracks:
                        tracks.append([(i+N/2, j+N/2)])
                        tracks_ceils_id.append((i/N, j/N))
    if len(tracks) > 0 and number_frame % DETECT_INTERVAL:
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(fgmask, old_frame, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(old_frame, fgmask, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            if not (x, y) == tr[-1]:
                tr.append((x, y))
            if len(tr) > 10:
                del tr[0]
            new_tracks.append(tr)
        tracks = new_tracks
        # print len(tracks)
        for tr in tracks:
            if len(tr) > 1:
                if not (float(tr[-2][0]) == float(tr[-1][0]) and float(tr[-2][1]) == float(tr[-1][1])):
                    movement += 1
                    # import pdb; pdb.set_trace()
                    x, y = tracks_ceils_id[tracks.index(tr)]
                    mot[x][y] = abs(np.linalg.norm(np.array(tr[-2])) - np.linalg.norm(np.array(tr[-1])))
                    cv2.circle(vis, (tr[-1][1], tr[-1][0]), N, (0, 255, 0))
        # import pdb; pdb.set_trace()

        # print number_frame
        mots.append(mot)

        draw_str(vis, (20, 20), 'movement detected: %d' % movement)
        movement = 0

    cv2.imshow('frame', vis)
    number_frame += 1
    old_frame = fgmask
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

pmfs = []
for i in range(len(mots)):
    avgMots.append(mots[i])
    divide = 1.0
    if i > 0:
        avgMots[i] += mots[i-1]
        divide += 1
    if i < len(mots) - 1:
        avgMots[i] += mots[i+1]
        divide += 1
    avgMots[i] = (np.array(avgMots[i], dtype='f')/divide).tolist()
    for speedArray in avgMots[i]:
        # import pdb; pdb.set_trace()
        kde = gaussian_kde(speedArray).pdf(range(len(speedArray)))
        pmfs.append(pmf.Pmf(kde))
        # pmfSpeed = pmfSpeed.__add__(pmf.Pmf(speedArray))

# import pdb; pdb.set_trace()
