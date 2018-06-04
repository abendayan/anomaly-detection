import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join
from scipy.stats.kde import gaussian_kde
import scipy.stats  as stats

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
class UCSDTrain:
    def __init__(self, path, n, detect_interval):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.tracks = []
        self.tracks_ceils_id = []
        self.n = n
        self.detect_interval = detect_interval
        self.data = []

    def create_cells(self, fgmask, width, height):
        ceils = [[None for x in range(height/self.n + 1)] for y in range(width/self.n + 1)]
        for i in range(0, width, self.n):
            for j in range(0, height, self.n):
                if np.any(fgmask[i:i+self.n,j:j+self.n] == 255):
                    # There is white in the mask
                    ceils[i/self.n][j/self.n] = fgmask[i:i+self.n,j:j+self.n]
                    toAdd = True
                    for tr in self.tracks:
                        if int(tr[-1][0]) <= i+self.n and int(tr[-1][0]) >= i and int(tr[-1][1]) <= j+self.n  and int(tr[-1][1]) >= j:
                            toAdd = False
                            break
                    if toAdd:
                        self.tracks.append([(i+self.n/2, j+self.n/2)])
                        self.tracks_ceils_id.append((i/self.n, j/self.n))
        return ceils

    def save_data(self, mots):
        j = 0
        for i in range(1, len(mots), 3):
            self.data.append(mots[i]+mots[i-1]+mots[i+1])
            self.data[j] = (np.array(self.data[j], dtype='f')/3.0).tolist()
            j += 1

    def learn_one_video(self, video_name):
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        files.remove('.DS_Store')
        files.remove('._.DS_Store')
        files.sort()
        number_frame = 0
        old_frame = None
        mots = []
        for tif in files:
            frame = cv2.imread(self.path + video_name + tif)
            fgmask = self.fgbg.apply(frame)
            width = fgmask.shape[0]
            height = fgmask.shape[1]
            frameCopy = frame.copy()

            mot = [[0 for x in range(height/self.n + 1)] for y in range(width/self.n + 1)]
            if number_frame % self.detect_interval == 0:
                self.tracks = []
                self.tracks_ceils_id = []
            ceils = self.create_cells(fgmask, width, height)
            if len(self.tracks) > 0 and number_frame % self.detect_interval:
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(fgmask, old_frame, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(old_frame, fgmask, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    if not (x, y) == tr[-1]:
                        tr.append((x, y))
                        if len(tr) > 10:
                            del tr[0]
                        new_tracks.append(tr)
                        self.tracks = new_tracks
                        movement = 0
                        for tr in self.tracks:
                            if len(tr) > 1:
                                if not (float(tr[-2][0]) == float(tr[-1][0]) and float(tr[-2][1]) == float(tr[-1][1])):
                                    x, y = self.tracks_ceils_id[self.tracks.index(tr)]
                                    mot[x][y] = abs(np.linalg.norm(np.array(tr[-2])) - np.linalg.norm(np.array(tr[-1])))
                                    cv2.circle(frameCopy, (tr[-1][1], tr[-1][0]), self.n, (0, 255, 0))
                                    mots.append(mot)
                                    movement += 1

                                    draw_str(frameCopy, (20, 20), 'movement detected: %d' % movement)
            cv2.imshow('frame', frameCopy)
            number_frame += 1
            old_frame = fgmask
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        self.save_data(mots)

if __name__ == '__main__':
    ucsd_training = UCSDTrain('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/', 16, 10)
    ucsd_training.learn_one_video('Train001/')

#
# # pmfs = []
# for i in range(len(mots)):
#     avgMots.append(mots[i])
#     divide = 1.0
#     if i > 0:
#         avgMots[i] += mots[i-1]
#         divide += 1
#     if i < len(mots) - 1:
#         avgMots[i] += mots[i+1]
#         divide += 1
#     avgMots[i] = (np.array(avgMots[i], dtype='f')/divide).tolist()
#     # print i
#     for speedArray in avgMots[i]:
#         pass
