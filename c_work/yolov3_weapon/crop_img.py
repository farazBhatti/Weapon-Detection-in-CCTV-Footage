import cv2

x = 60
y = 20
w = 219 * 1.1
h = 146 * 1.1


img = cv2.imread("test_data/imgs/gun1.jpeg")
crop_img = img[y:int(y+h), x:int(x+w)]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()