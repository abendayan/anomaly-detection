import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats.kde import gaussian_kde
import scipy.stats  as stats
from scipy.stats import norm
from sklearn import svm
import matplotlib.pyplot as plt
import random

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
class Classifier:
    def __init__(self):
        self.T = 3.0
        self.N = 0

    def train_classifier(self, data):
        for cells in data:
            prob = []
            for cell in cells:
                if not np.all(np.array(cell) == 0):
                    density = gaussian_kde(cell)
                    density.covariance_factor = lambda : .25
                    density._compute_covariance()
                    prob.append((density.pdf(range(len(cell))).sum())/(self.N))
            if len(prob) > 0:
                if min(prob) < self.T:
                    self.T = min(prob)

    def is_anomaly(self, cells):
        anomaly = [False]*len(cells)
        i = 0
        for cell in cells:
            if not np.all(np.array(cell) == 0):
                density = gaussian_kde(cell)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                if (density.pdf(range(len(cell))).sum())/(self.N) < self.T:
                    anomaly[i]= True
            i += 1
        return anomaly


class UCSD:
    def __init__(self, path, n, detect_interval, train = True):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.tracks = []
        self.tracks_ceils_id = []
        self.n = n
        self.detect_interval = detect_interval
        self.data = []
        self.train = train

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
            if i+1 < len(mots):
                nextOne = mots[i+1]
                divide = 3.0
            else:
                divide = 2.0
            self.data.append(mots[i])
            self.data[j] = (np.array(self.data[j], dtype='f')/divide).tolist()
            j += 1

    def learn_one_video(self, video_name, classifier = None):
        self.data = []
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if '._.DS_Store' in files:
            files.remove('._.DS_Store')
        files.sort()
        number_frame = 0
        old_frame = None
        mots = []
        for tif in files:
            movement = 0
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

                        if len(tr) > 1:
                            if not (float(tr[-2][0]) == float(tr[-1][0]) and float(tr[-2][1]) == float(tr[-1][1])):
                                x, y = self.tracks_ceils_id[self.tracks.index(tr)]
                                mot[x][y] = abs(np.linalg.norm(np.array(tr[-2])) - np.linalg.norm(np.array(tr[-1])))
                if not self.train:
                    is_anomaly = speedClassifier.is_anomaly(mot)
                    if sum(is_anomaly) > 0:
                        index = 0
                        really_anomaly = False
                        for anomaly in is_anomaly:
                            if anomaly:
                                if (index > 0 and is_anomaly[index-1]) or (index < len(is_anomaly) - 1 and is_anomaly[index+1]):
                                    really_anomaly = True
                                    tr = self.tracks[index]
                                    cv2.circle(frameCopy, (tr[-1][1], tr[-1][0]), self.n, (0, 255, 0))
                                    draw_str(frameCopy, (20, 20), 'anomaly detected:')
                            index += 1
                        if really_anomaly:
                            print number_frame
            if self.train:
                mots.append(mot)
            cv2.imshow('frame', frameCopy)
            number_frame += 1
            old_frame = fgmask
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        if self.train:
            self.save_data(mots)

if __name__ == '__main__':
    ucsd_training = UCSD('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/', 16, 10)
    dir_trains = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/', f))]
    dir_trains.sort()
    speedClassifier = Classifier()
    ucsd_training.learn_one_video('Train001/')
    speedClassifier.N = len(ucsd_training.data)*len(ucsd_training.data[0])*len(ucsd_training.data[0][0])
    # speedClassifier.N = 10050
    speedClassifier.train_classifier(ucsd_training.data)

    dir_trains.pop(0)

    for directory in dir_trains:
        print directory
        ucsd_training.learn_one_video(directory+'/')
        speedClassifier.train_classifier(ucsd_training.data)

    print speedClassifier.T

    # speedClassifier.T = 0.0002573289882909553

    ucsd_testing = UCSD('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/', 16, 10, False)
    ucsd_testing.learn_one_video('Test002/', speedClassifier)
