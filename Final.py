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
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 1)
        imgContour = draw_circle(imgContour, contours)
    return imgContour
def draw_circle(img, contours):
    cir = np.zeros_like(img)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.12 * peri, True)
        (x, y), radius = cv2.minEnclosingCircle(approx)
        center = (int(x), int(y))
        radius = int(radius)
        radius = radius+2
        cv2.circle(cir, center, radius, (255, 0, 0), -1)
    return cir

def findContours2(img):
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

def preprocess(img):
    imgt = img.copy()
    imgt = increase_saturation(imgt,127)
    # imgt = increase_contrast(imgt, 0.6)
    imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
    imgt = cv2.GaussianBlur(imgt, (3, 3), 0)
    imgt = cv2.Canny(imgt, 0, 100)
    imgt = findContours2(imgt)

    imgt = cv2.bitwise_and(img,img,mask=imgt)

    imgt = cv2.resize(imgt, (200,200))
    return imgt










paths = ["..\character-segmentation\Test_images\{}.jpeg".format(i) for i in range(1, 8)]
pic = [None] * 8
for i in range(1, 8):
    pic[i] = cv2.imread(paths[i-1])
    pic[i] = preprocess(pic[i])








blank = np.zeros_like(pic[7])
stackedH1 = np.hstack((pic[1], pic[2], pic[3]))
stackedH2 = np.hstack((pic[4], pic[5], pic[6]))
stackedH3 = np.hstack((pic[7], blank, blank))
result = np.vstack((stackedH1, stackedH2, stackedH3))
cv2.imshow("Result", result)
# cv2.imwrite("C:\\Users\Theodore Regimon\Desktop\\test0.png", result)
cv2.waitKey(0)
