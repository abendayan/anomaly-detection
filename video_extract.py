import sys
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join, isdir
import random

class UCSD:
    def __init__(self, path, n, detect_interval):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.n = n
        self.detect_interval = detect_interval

    def process_frame(self, bins, magnitude, fmask, out, tag_image = None):
        bin_count = np.zeros(9, np.uint8)
        h,w, t = bins.shape
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
                if tag_image is None:
                    tag = 0
                else:
                    tag_atom = tag_image[i:i_end, j:j_end].flatten()
                    ones = np.count_nonzero(tag_atom)
                    zeroes = len(tag_atom) - ones
                    tag = 1
                    # print ones
                    if ones < 50:
                        tag = 0
                features = hs.tolist()
                features.extend([f_cnt, atom_mag, tag])
                for f in features:
                    out.write(str(f) + " ")
                out.write("\n")
        return 0

    def extract_features(self, video_name, type, tag_video = ""):
        mag_threshold=1e-3
        elements = 0
        is_tagged = not tag_video == ""
        out = open("features/features_test_"+type+"_"+video_name.split("/")[0]+".txt","w")
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        if is_tagged:
            files_tag = [f for f in listdir(self.path+tag_video) if isfile(join(self.path+tag_video, f))]
            if '.DS_Store' in files_tag:
                files_tag.remove('.DS_Store')
            if '._.DS_Store' in files_tag:
                files_tag.remove('._.DS_Store')
            files_tag.sort()
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
        if is_tagged:
            tag_img = np.zeros((h,w,self.n), np.uint8)
        for tif in files:
            movement = 0
            frame = cv2.imread(self.path + video_name + tif, cv2.IMREAD_GRAYSCALE)
            fmask[...,number_frame % self.detect_interval] = self.fgbg.apply(frame)
            frameCopy = frame.copy()
            flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if is_tagged:
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
                if is_tagged:
                    self.process_frame(bins, mag, frameCopy, out, tag_img)
                else:
                    self.process_frame(bins, mag, frameCopy, out)
            cv2.imshow('frame', frameCopy)
            number_frame += 1
            old_frame = frame
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        out.close()

if __name__ == '__main__':
    ucsdped = 'UCSDped1'
    ucsd_training = UCSD('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/', 10, 5)
    dir_trains = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Train/', f))]
    dir_tests = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', f))]
    dir_trains.sort()
    # out = open("features_test_UCSDped1.txt","w")
    # ucsd_training.extract_features('Train001/', ucsdped)
    dir_trains.pop(0)

    li = ["Test/Test003","Test/Test004","Test/Test014","Test/Test018","Test/Test019", "Test/Test021","Test/Test022","Test/Test023","Test/Test024","Test/Test032"]
    ucsd_training.path = 'UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/'
    for directory in li:
        print directory
        if not directory.endswith("gt"):
            dir_split = directory.split("/")[1]
            dir_split = dir_split + '_gt'
            if dir_split in dir_tests:
                ucsd_training.extract_features(directory+'/', ucsdped, directory + '_gt/')
            else:
                ucsd_training.extract_features(directory+'/', ucsdped)
