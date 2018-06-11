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
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import random

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

lk_params = dict( winSize  = (238,158),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def flatten(l):
    return [item for sublist in l for item in sublist]

class Classifier:
    def __init__(self):
        self.T = 3.0
        self.N = 0
        self.prob = []
        self.all_data = []

    def train_classifier(self, data):
        all_data = []
        for x in range(len(data)):
            # if x >= len(self.prob):
            #     self.prob.append([])
            for y in range(len(data[0])):
                # if y >= len(self.prob[x]):
                #     self.prob[x].append(3)
                # if len(data[x][y]) > 1:
                #     density = gaussian_kde(data[x][y])
                #     density.covariance_factor = lambda : .25
                #     density._compute_covariance()
                #     pmf = (density.pdf(range(len(data[x][y]))).sum())/(self.N)
                #     if self.prob[x][y] > pmf:
                #         self.prob[x][y] = pmf
                all_data.extend(data[x][y])
        self.all_data.extend(all_data)

        # import pdb; pdb.set_trace()
        # # self.all_data.extend(all_data)
        # if self.T > pmf.sum()/self.N:
        #     self.T = pmf.sum()/self.N

    def optimize(self):
        self.density = gaussian_kde(self.all_data)
        self.density.covariance_factor = lambda : .25
        self.density._compute_covariance()
        pmf = self.density.pdf(range(len(self.all_data)))
        self.T = pmf.sum()/self.N
        self.all_data = []

    def is_anomaly(self, cell, x, y):
        pmf = self.density(cell)[0]/self.N
        if pmf < self.T:
            print pmf
            return True
        return False

class UCSD:
    def __init__(self, path, n, detect_interval, train = True, fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()):
        self.path = path
        self.fgbg = fgbg
        self.tracks = []
        self.tracks_ceils_id = []
        self.n = n
        self.detect_interval = detect_interval
        self.data = []
        self.train = train
        self.index_tracks = []

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
        elements = 0
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if '._.DS_Store' in files:
            files.remove('._.DS_Store')
        files.sort()
        number_frame = 0
        old_frame = None
        mots = []
        frame = cv2.imread(self.path + video_name + '001.tif')
        width = frame.shape[0]
        height = frame.shape[1]
        mot = [[[] for x in range(height/self.n + 1)] for y in range(width/self.n + 1)]
        for tif in files:
            movement = 0
            frame = cv2.imread(self.path + video_name + tif)
            fgmask = self.fgbg.apply(frame)
            frameCopy = frame.copy()

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

                        if len(tr) > 1 and not (float(tr[-2][0]) == float(tr[-1][0]) and float(tr[-2][1]) == float(tr[-1][1])):
                                x, y = tr[-1]
                                x = int(x/self.n)
                                y = int(y/self.n)
                                mot[x][y].append(abs(np.linalg.norm(np.array(tr[-2])) - np.linalg.norm(np.array(tr[-1]))))
                                elements += 1
                                if not self.train:
                                    is_anomaly = speedClassifier.is_anomaly(mot[x][y][-1], x, y)
                                    if is_anomaly:
                                        import pdb; pdb.set_trace()
                                        print number_frame, tr[-1]
                                        cv2.circle(frameCopy, (tr[-1][1], tr[-1][0]), self.n, (0, 255, 0))
                                        draw_str(frameCopy, (20, 20), 'anomaly detected:')
            cv2.imshow('frame', frameCopy)
            number_frame += 1
            old_frame = fgmask
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        if self.train:
            # self.save_data(mot)
            self.data = mot

if __name__ == '__main__':
    ucsdped = 'UCSDped1'
    ucsd_training = UCSD('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/', 16, 2)
    dir_trains = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/', f))]
    dir_trains.sort()
    speedClassifier = Classifier()
    ucsd_training.learn_one_video('Train001/')
    speedClassifier.N = len(ucsd_training.data)*len(ucsd_training.data[0])
    # speedClassifier.N = 150
    speedClassifier.train_classifier(ucsd_training.data)

    dir_trains.pop(0)

    # for directory in dir_trains:
    #     print directory
    #     ucsd_training.learn_one_video(directory+'/')
    #     speedClassifier.train_classifier(ucsd_training.data)

    speedClassifier.optimize()
    # print speedClassifier.s
    # print speedClassifier.m
    # speedClassifier.s = 0.48227979951955857
    # speedClassifier.m = 0.4558405032395014
    # speedClassifier.min = speedClassifier.m - 3*speedClassifier.s
    # speedClassifier.max = speedClassifier.m + 3*speedClassifier.s
    # print speedClassifier.min, speedClassifier.max
    print speedClassifier.T
    # speedClassifier.T = 0.0063071457639960835


    # ucsd_testing = UCSD('UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/', 16, 2, False)
    ucsd_testing = UCSD('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', 16, 2, False, ucsd_training.fgbg)
    ucsd_testing.learn_one_video('Test001/', speedClassifier)
