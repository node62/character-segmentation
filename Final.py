import cv2
import numpy as np
path = "..\character-segmentation\Test_images\\1.jpeg"

def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    imgCanny = cv2.Canny(imgBlur, 280, 300)
    return imgCanny

def contours(img):
    biggest=np.array([])
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    maxArea = 0
    imgContours = np.zeros_like(img)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > maxArea:
            biggest = approx
            maxArea = area
    cv2.drawContours(imgContours, contours, -1, (255, 0, 0), -1)
    return imgContours

#################################################
img = cv2.imread(path)
img = preprocess(img)

result = contours(img)
result = cv2.resize(result, (512, 512))
cv2.imshow("Result", result)
cv2.waitKey(0)