import numpy as np
import cv2

i = 3
path = 'D:\\training_data\human data for classification\Human\\'

img1 = cv2.imread(path + '11 (2).jpg', 0)

img2 = cv2.resize(img1, dsize=(64,64))

cv2.imshow(winname='img2', mat=img2)

cv2.imwrite('halo' + str(i) + '.jpg',img2)

cv2.waitKey(0)
cv2.destroyWindow()


