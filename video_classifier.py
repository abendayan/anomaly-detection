import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join, isdir
import random
from sklearn.tree import DecisionTreeClassifier
from UCSDped1 import TestVideoFile
from sklearn.neighbors import KNeighborsClassifier
from model import VideoCLassifier
import time

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

class UCSDTest:
    def __init__(self, path, n, detect_interval, type):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.n = n
        self.detect_interval = detect_interval
        # self.clf = DecisionTreeClassifier(max_depth=5)
        # self.clf = KNeighborsClassifier(3)
        # self.load_train_features(type)
        self.classifier = VideoCLassifier()
        self.correct = 0.0
        self.found = 0.0
        self.should_find = 0.0

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
        # self.clf.fit(x_train, y_train)

    def process_frame(self, bins, magnitude, fmask, tag_img):
        bin_count = np.zeros(9, np.uint8)
        h,w, t = bins.shape
        found_anomaly = False
        predicted = [ [0 for j in range(0, w, self.n)] for i in range(0, h, self.n) ]
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
                features = hs.tolist()
                features.extend([f_cnt, atom_mag, i, j])
                # vector = np.array(features)
                tag_atom = tag_img[i:i_end, j:j_end].flatten()
                ones = np.count_nonzero(tag_atom)
                zeroes = len(tag_atom) - ones
                tag = 1
                # print ones
                if ones < 50:
                    tag = 0
                predicted = self.classifier.predict(features, tag)
                if tag == 1:
                    self.should_find += 1
                if predicted == tag and predicted == 1:
                    self.correct += 1
                if predicted == 1:
                    self.found += 1
                    cv2.rectangle(fmask, (j, i), (j_end, i_end), (255,0,0), 2)
                    found_anomaly = True
        return found_anomaly

    def process_video(self, video_name, tag_video):
        mag_threshold=1e-3
        elements = 0
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if '._.DS_Store' in files:
            files.remove('._.DS_Store')
        files_tag = [f for f in listdir(self.path+tag_video) if isfile(join(self.path+tag_video, f))]
        if '.DS_Store' in files_tag:
            files_tag.remove('.DS_Store')
        if '._.DS_Store' in files_tag:
            files_tag.remove('._.DS_Store')
        files_tag.sort()
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
        tag_img = np.zeros((h,w,self.n), np.uint8)
        anomaly_detected = []
        for tif in files:
            movement = 0
            frame = cv2.imread(self.path + video_name + tif, cv2.IMREAD_GRAYSCALE)
            fmask[...,number_frame % self.detect_interval] = self.fgbg.apply(frame)
            frameCopy = frame.copy()
            flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            tag_img_ = cv2.imread(self.path + tag_video + files_tag[number_frame] ,cv2.IMREAD_GRAYSCALE)
            tag_img[...,number_frame % self.detect_interval] = tag_img_
            # Calculate direction and magnitude
            height, width = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]
            angle = ((np.arctan2(fy, fx+1) + 2*np.pi)*180)% 360
            binno = np.ceil(angle/45)
            magnitude = np.sqrt(fx*fx+fy*fy)
            binno[magnitude < mag_threshold] = 0
            bins[...,number_frame % self.detect_interval] = binno
            mag[..., number_frame % self.detect_interval] = magnitude
            if number_frame % self.detect_interval == 0:
                found_anomaly = self.process_frame(bins, mag, frameCopy, tag_img)
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
    ucsd_test = UCSDTest('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', 10, 1, ucsdped)
    dir_test = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', f))]
    dir_test.sort()
    total_correct = 0.0
    total_should_found = 0.0
    total_found = 0.0
    for directory in dir_test:
        print directory
        if not directory.endswith("gt"):
            start_time = time.time()
            anomaly_detected = ucsd_test.process_video(directory+'/', directory + '_gt/')
            time_video = passed_time(start_time)
            print 200.0/time_video, "frames per second"
            total_found += len(anomaly_detected)
            index_video = int(directory[-3:])
            total_correct += len(set(anomaly_detected).intersection(TestVideoFile[index_video]))
            total_should_found += len(TestVideoFile[index_video])

    total_correct_pixel = ucsd_test.correct
    total_should_found_pixel = ucsd_test.should_find
    total_found_pixel = ucsd_test.found
    precision = total_correct/total_found
    recall = total_correct/total_should_found
    f1 = 2.0*precision*recall/(precision+recall)
    print "Results frame wise:"
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1
    print "Results pixel wise:"
    precision = total_correct_pixel/total_found_pixel
    recall = total_correct_pixel/total_should_found_pixel
    f1 = 2.0*precision*recall/(precision+recall)
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1
