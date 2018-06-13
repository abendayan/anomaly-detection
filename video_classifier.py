import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join, isdir
import random
from sklearn.tree import DecisionTreeClassifier
from UCSDped1 import TestVideoFile

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

class UCSDTest:
    def __init__(self, path, n, detect_interval, type):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.n = n
        self.detect_interval = detect_interval
        self.clf = DecisionTreeClassifier(max_depth=5)
        self.load_train_features(type)

    def load_train_features(self, type):
        x_train = []
        y_train = []
        features = [f for f in listdir('features/') if f.startswith("features_test_"+type)]
        for feature in features:
            file = open('features/' + feature, "r")
            feature_text = file.read().split("\n")
            for f in feature_text:
                if f!= "":
                    feat_all = [float(feat) for feat in f.split(" ")[:-1]]
                    x_train.append(feat_all[:-1])
                    y_train.append(int(feat_all[-1]))
        self.clf.fit(x_train, y_train)

    def process_frame(self, bins, magnitude, fmask):
        bin_count = np.zeros(9, np.uint8)
        h,w, t = bins.shape
        found_anomaly = False
        for i in range(0, h, self.n):
            for j in range(0, w, self.n):
                i_end = min(h, i+self.n)
                j_end = min(w, j+self.n)

                # Get the atom for bins
                atom_bins = bins[i:i_end, j:j_end].flatten()

                # Average magnitude
                atom_mag = magnitude[i:i_end, j:j_end].flatten().mean()
                atom_fmask = fmask[i:i_end, j:j_end].flatten()

                # Count of foreground values
                f_cnt = np.count_nonzero(atom_fmask)

                # Get the direction bins values
                hs, _ = np.histogram(atom_bins, np.arange(10))

                # get the tag atom
                # tag_atom = tag_image[i:i_end, j:j_end].flatten()
                #print(tag_atom)
                # ones = np.count_nonzero(tag_atom)
                # zeroes = len(tag_atom) - ones
                tag = 0
                # if ones < self.n:
                    # tag = 0
                features = hs.tolist()
                features.extend([f_cnt, atom_mag])
                vector = np.matrix(features)
                predicted = self.clf.predict(vector)[0]
                if predicted == 1:
                    # import pdb; pdb.set_trace()
                    cv2.rectangle(fmask, (j, i), (j_end, i_end), (255,0,0), 2)
                    found_anomaly = True
                    # cv2.circle(fmask, (j, i), self.n, (0, 255, 0))
                # out.write()
                # print(f, end=",", file=out)
                # print("\n", end="", file=out)
        return found_anomaly

    def process_video(self, video_name):
        mag_threshold=1e-3
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
        old_frame = cv2.imread(self.path + video_name + '001.tif', cv2.IMREAD_GRAYSCALE)
        width = old_frame.shape[0]
        height = old_frame.shape[1]
        h, w = old_frame.shape[:2]
        bins = np.zeros((h, w, self.detect_interval), np.uint8)
        mag = np.zeros((h, w, self.detect_interval), np.float32)
        fmask = np.zeros((h, w, self.detect_interval), np.uint8)
        frames = np.zeros((h, w, self.detect_interval), np.uint8)
        anomaly_detected = []
        for tif in files:
            movement = 0
            frame = cv2.imread(self.path + video_name + tif, cv2.IMREAD_GRAYSCALE)
            fmask[...,number_frame % self.detect_interval] = self.fgbg.apply(frame)
            frameCopy = frame.copy()
            flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Calculate direction and magnitude
            height, width = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]
            angle = ((np.arctan2(fy, fx+1) + 2*np.pi)*180)% 360
            binno = np.ceil(angle/45)
            magnitude = np.sqrt(fx*fx+fy*fy)
            binno[magnitude < mag_threshold] = 0
            bins[...,number_frame % self.detect_interval] = binno
            mag[..., number_frame % self.detect_interval] = magnitude
            # if number_frame % self.detect_interval == 0:
            found_anomaly = self.process_frame(bins, mag, frameCopy)
            if found_anomaly:
                anomaly_detected.append(number_frame)
            cv2.imshow('frame', frameCopy)
            number_frame += 1
            old_frame = frame
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        return anomaly_detected

if __name__ == '__main__':
    ucsdped = 'UCSDped1'
    ucsd_test = UCSDTest('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', 10, 5, ucsdped)
    dir_test = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', f))]
    dir_test.sort()
    total_correct = 0.0
    total_should_found = 0.0
    total_found = 0.0
    for directory in dir_test:
        print directory
        if not directory.endswith("gt"):
            anomaly_detected = ucsd_test.process_video(directory+'/')
            total_found += len(anomaly_detected)
            index_video = int(directory[-3:])
            total_correct += len(set(anomaly_detected).intersection(TestVideoFile[index_video]))
            total_should_found += len(TestVideoFile[index_video])

    precision = total_correct/total_found
    recall = total_correct/total_should_found
    f1 = 2.0*precision*recall/(precision+recall)
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1
