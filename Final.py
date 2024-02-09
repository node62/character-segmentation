import cv2
import numpy as np
kernel = np.ones((3, 3), np.uint8)

def increase_saturation(img, s_min = 0, s_max = 255, v_min = 0, v_max = 255, h_min = 0, h_max = 179):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)
    return imgResult
def findContours(img):
    imgContour = np.zeros_like(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 1)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (255, 0, 0), -1)
    return imgContour
def kmeans_segmentation(image, k):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image
def find_least_frequent_pixels(image):
    flat_image = image.flatten()
    unique_values, counts = np.unique(flat_image, return_counts=True)
    least_frequent_value = unique_values[np.argmin(counts)]
    least_frequent_mask = (image == least_frequent_value).astype(np.uint8)
    return least_frequent_mask * 255
def white(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mat = np.array(img)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j]!=0:
                mat[i][j]=255
    return mat
def preprocess(img):
    imgt = img.copy()
    imgt = increase_saturation(imgt,128)
    imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
    imgt = cv2.GaussianBlur(imgt, (3, 3), 0)
    imgt = cv2.Canny(imgt, 0, 100)
    imgt = findContours(imgt)
    imgt = cv2.bitwise_and(img,img,mask=imgt)
    imgt = kmeans_segmentation(imgt, 4)
    imgt = find_least_frequent_pixels(imgt)
    imgt = white(imgt)
    imgt = cv2.bitwise_and(img,img,mask=imgt)
    return imgt
##########################################################
paths = ["..\character-segmentation\Test_images\{}.jpeg".format(i) for i in range(1, 8)]
pic = [None] * 8
for i in range(1, 8):
    pic[i] = cv2.imread(paths[i-1])
    pic[i] = preprocess(pic[i])
    cv2.imwrite(f"..\character-segmentation\segmented_test_images\\result{i}.png", pic[i])

blank = np.zeros_like(pic[7])
stackedH1 = np.hstack((pic[1], pic[2], pic[3]))
stackedH2 = np.hstack((pic[4], pic[5], pic[6]))
stackedH3 = np.hstack((pic[7], blank, blank))
result = np.vstack((stackedH1, stackedH2, stackedH3))

cv2.imshow("Result", result)
cv2.imwrite("result.png", result)
cv2.waitKey(0)
