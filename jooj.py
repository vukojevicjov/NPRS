import cv2 as cv

img = cv.imread("slike/put.jpg")

cv.putText(img, "AUTOMATIKA JE NAJBOLJA!!!", (img.shape[0]//2, img.shape[1]//6), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
cv.putText(img, "DA LI KASTELAN OVO VIDI???", (img.shape[0]//2, img.shape[1]//2), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

cv.imshow("img", img)

cv.waitKey(0)