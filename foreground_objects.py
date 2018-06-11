import numpy as np
import cv2
import os
import math
import glob

background = cv2.imread("background.tiff")
N=16
threshold_grayscale=35

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

kernel = np.ones((5,5),np.uint8)
for f in glob.glob(r'/Users/eva/Documents/AnomalyDetectionData/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001/*'):
    img = cv2.imread(f)
    foreground = get_foreground(f)
    #closing = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    #dilation = cv2.dilate(foreground, kernel, iterations=1)
    contours = get_contours(foreground)
    areas = []
    perimeters = []
    i = 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 0:
            areas.append(a)
        p = cv2.arcLength(cnt, True)
        if p > 0:
            perimeters.append(p)
        #if a > 20 and p > 0:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 1)

    filename = os.path.basename(f)
    cv2.imwrite('Test001-output/{}'.format(filename),img)
    print areas
    print 'average',sum(areas)/len(areas)
    print perimeters
    print 'average',sum(perimeters)/len(perimeters)


#cv2.drawContours(img, get_contours(foreground), -1, (0, 255, 0), 3)

