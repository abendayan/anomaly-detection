import numpy as np
import cv2
import os
import math
import glob
from model import VideoLearn
background = cv2.imread("background.tiff")
N=16
threshold_grayscale=30
reg0=25
reg1=50
reg2=100
regions_Y=[25,50,75,100,125,140]
areas=[20,30,40]
#BGR = blue green red yellow pink
colors=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def compute_size_in_cell(matrix, focus, row, col):
    sum = 0
    max_row = matrix.shape[0] - 1
    max_col = matrix.shape[1] - 1
    try:
        for i in xrange(row -1,row +2):
            for j in xrange(col-1, col+2):
                if 0 <= i <= max_row and 0 <= j <= max_col:
                    sum += matrix[i][j]
    except IndexError:
        print "error"
    sum += (focus-1) * matrix[row][col]
    return sum



def get_background():
    b_up =cv2.imread('/Users/eva/Documents/AnomalyDetectionData/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train011/200.tif')
    b_down=cv2.imread('/Users/eva/Documents/AnomalyDetectionData/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train010/200.tif')
    new_up = b_up[0:73]
    new_down = b_down[73:]
    return np.concatenate((new_up,new_down))

def get_foreground(img_name):
    img = cv2.imread(img_name)
    foreground = cv2.absdiff(img, background)
    img_name = os.path.basename(img_name)
    cv2.imwrite('Test001-foreground/{}'.format(img_name), foreground)
    img_grey = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_grey, threshold_grayscale, 255, cv2.THRESH_BINARY)
    cv2.imwrite('Test001-thresh/{}'.format(img_name), thresh)
    return thresh

def get_cells_pixels(img):
    width = img.shape[0]
    height = img.shape[1]

    cell_matrix_shape = (int(math.ceil(float(width) / N)), int(math.ceil(float(height) / N)))
    foreground_pixels_in_frame = np.zeros((cell_matrix_shape[1], cell_matrix_shape[0]))

    a = b = 0
    for i in range(0, height , N):
        for j in range(0, width, N):
            current_cell = img[i:min(i + N,height-1), j:min(j + N,width-1)]
            number_of_foreground_pixels = np.count_nonzero(current_cell == 255) # count number of white pixels in cell
            foreground_pixels_in_frame[a][b] = number_of_foreground_pixels
            b += 1
        a += 1
        b = 0

    sizes_in_cells = np.zeros((cell_matrix_shape[1], cell_matrix_shape[0]))
    focus = 2
    # current frame computation
    for x in xrange(0, cell_matrix_shape[1]):
        for y in xrange(0,cell_matrix_shape[0]):
            sizes_in_cells[x][y] = compute_size_in_cell(foreground_pixels_in_frame, focus, x, y)


def get_contours(thresh):
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def sort_contours_by_region(contours):
    contour_in_region_i = []
    for i in xrange(0,5):
        contour_in_region_i.append([])
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        bounding_rect_area = w*h
        # if cv2.contourArea(c) < 2:
        #     continue
        l = None # find region according to y
        if y < 25 or y > 140: # do not take into considerations objects with y<25 or y>140
            continue
        elif 25 <= y <= 50:
            l = contour_in_region_i[0]
        elif 50 < y <= 75:
            l = contour_in_region_i[1]
        elif 75 < y <= 100:
            l = contour_in_region_i[2]
        elif 100 < y <= 125:
            l = contour_in_region_i[3]
        elif 125 < y <= 140:
            l = contour_in_region_i[4]

        l.append((c, (x,y,w,h)))

    return contour_in_region_i
    # for i in xrange(0,6):
    #     print "region ",i
    #     contour_in_region_i[i]
    #     print "average height:"

def most_common(lst):
    return max(set(lst), key=lst.count)

def most_common_height(lst):
    l = [x[1][3] for x in lst] # get height in each element of lst
    return most_common(l)

def average(l):
    return sum(l) / float(len(l))

def average_height(lst):
    l = [x[1][3] for x in lst]  # get height in each element of lst
    return average(l)

def find_humans_according_to_area(folder,img,img_name,regions):
    for i in xrange(0,5):
        if not regions[i]:
            return
        # default_height_for_region = average_height(regions[i])
        # print "default region",default_height_for_region
        for tup in regions[i]:
            cnt = tup[0]
            a = cv2.contourArea(cnt)
            x, y, w, h = tup[1]
            if a > 6:
                cv2.drawContours(img, [cnt], 0, colors[i], 1)
                tags[i].append(get_tag(folder,img_name,x,y,w,h))
                heights_in_good_contours[i].append(h)
                widths_in_good_contours[i].append(w)
                perimeters[i].append(cv2.arcLength(cnt, True))
                areas[i].append(a)
                print "Region ", i ,"height",h, "width", w
            else:
                print "area",a,"x",x,"y",y,"height",h, "width", w

    # for i in xrange(0,3):
    #     for tup in regions[i]:
    #         cnt = tup[0]
    #         a = cv2.contourArea(cnt)
    #         bounding_r_area = tup[1]
    #         if a > areas[i]:
    #             cv2.drawContours(img, [cnt], 0, colors[i], 1)
    #             print "IN",i, "ratio: ",a/float(bounding_r_area)
    #             if i == 1 and "133" in img_name:
    #                 x, y, w, h = cv2.boundingRect(cnt)
    #                 print x, y, w, h
    #                 cv2.imwrite('tttt.tiff'.format(filename), img)
    #         else:
    #             cv2.drawContours(img, [cnt], 0, (255,255,255), 1)
    #             print "OUT",i,tup[1],a


def get_tag(folder,img_name,x,y,w,h):
    frame_num=img_name.split(".")[0]
    # check if any of the pixels within bounding rectangle is white - if so, tag as 1
    # else tag as 0
    groundtruth_img_name = '/Users/eva/Documents/AnomalyDetectionData/' \
                      'UCSD_Anomaly_Dataset.v1p2/UCSDped1/{}_gt/{}.bmp'.format(folder,frame_num)
    groundtruth_img = cv2.imread(groundtruth_img_name)
    for i in xrange(y,y+h):
        for j in xrange(x,x+w):
            pixel_color = groundtruth_img[i][j]
            if np.array_equal(pixel_color,[255,255,255]):
                return 1
    return 0

def write_tag(t):
    outfile="tags"
    np.save(outfile, t)
    with open(outfile + ".npy","r") as f:
        f.seek(0)

def write_features(features):
    outfile="features.txt"
    import json
    with open(outfile, 'w') as f:
        json.dump(features, f)


kernel = np.ones((2,2),np.uint8)
heights_in_good_contours = []
widths_in_good_contours = []
perimeters = []
areas = []
tags = []
for i in xrange(0, 5):
    widths_in_good_contours.append([])
    heights_in_good_contours.append([])
    tags.append([])
    perimeters.append([])
    areas.append([])

train_folders = ["Test/Test003", "Test/Test004", "Test/Test014", "Test/Test018", "Test/Test019", "Test/Test021", "Test/Test022",
      "Test/Test023", "Test/Test024", "Test/Test032"]

for folder in train_folders:
    for f in glob.glob(r'/Users/eva/Documents/AnomalyDetectionData/UCSD_Anomaly_Dataset.v1p2/'
                       r'UCSDped1/{}/*'.format(folder)):
        img = cv2.imread(f)
        img_name = os.path.basename(f)
        foreground = get_foreground(f)
        #closing = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        #dilation = cv2.dilate(foreground, kernel, iterations=1)
        contours = get_contours(foreground)
        find_humans_according_to_area(folder,img,img_name,sort_contours_by_region(contours))

        # areas = []
        # perimeters = []
        # i = 0
        # for cnt in contours:
        #     a = cv2.contourArea(cnt)
        #     if a > 0:
        #         areas.append(a)
        #     p = cv2.arcLength(cnt, True)
        #     if p > 0:
        #         perimeters.append(p)
        #     #if a > 20 and p > 0:
        #     cv2.drawContours(img, [cnt], 0, (255, 0, 0), 1)
        #
        #filename = os.path.basename(f)
        #cv2.imwrite('Test003-output/{}'.format(filename),img)
        # print areas
        # print 'average',sum(areas)/len(areas)
        # print perimeters
        # print 'average',sum(perimeters)/len(perimeters)


# merge all lists
import itertools
all_heights = list(itertools.chain.from_iterable(heights_in_good_contours))
all_widths = list(itertools.chain.from_iterable(widths_in_good_contours))
all_areas = list(itertools.chain.from_iterable(areas))
all_perimeters = list(itertools.chain.from_iterable(perimeters))
all_tags = list(itertools.chain.from_iterable(tags))

features = map(lambda x,y,w,z:[x,y,w,z],all_heights,all_widths,all_areas,all_perimeters)
write_features(features)
write_tag(np.array(all_tags))

