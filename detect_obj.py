import cv2 
import matplotlib.pyplot as plt
import numpy as np



# read images
img1 = cv2.imread('biscuits_packets_train_sift.jpg')  
img2 = cv2.imread('lays_test_sift.jpg') 

 
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR5552GRAY)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
#Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm by D.
sift = cv2.xfeatures2d.SIFT_create()



ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()


keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#Printing the keypoint

print(len(keypoints_1), len(keypoints_2))


#feature matching
#Brute force Matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


matches = bf.match(descriptors_1,descriptors_2)
print(len(matches))

matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()


