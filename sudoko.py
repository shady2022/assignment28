import argparse
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours
from imutils.perspective import four_point_transform as im

parser = argparse.ArgumentParser(description="Sudoku Detector version 1.0")

parser.add_argument("--input" , type=str , help="path of your input image")
parser.add_argument("--filter_size" , type=int , help="size of GaussianBlur mask", default= 7)
parser.add_argument('--final-size', type=int, help='Size of final sudoku image', default=700)
parser.add_argument("--output" , type=str , help="path of your output image")


args = parser.parse_args()

img = cv2.imread("D:\\Python Project\\python_programming\\tamrin8\\sudoku1.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_img = cv2.GaussianBlur(gray_img , (args.filter_size,args.filter_size), 3)
#cv2.imshow("blurred_img", blurred_img)

thresh = cv2.adaptiveThreshold(blurred_img , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV , 11 ,2)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours , key = cv2.contourArea , reverse = True)

sudoku_contour = None

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour , True)
    approx = cv2.approxPolyDP(contour , epsilon , True)
                              
    if len(approx) == 4:
        sudoku_contour = approx
        break

if sudoku_contour is None :
    print("Not found")
else:
    result = cv2.drawContours(img , [sudoku_contour] , -1 ,(0,255,0) , 8)
    sudoko_img = im(img, sudoku_contour.reshape(4,2))
    warped = im(gray_img, sudoku_contour.reshape(4,2))
    cv2.imshow("sudouko" ,sudoko_img)
    #cv2.imwrite("args.output", sudoko_img)


plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(blurred_img ,cmap="gray")
plt.title("blurred Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(thresh ,cmap="gray") 

plt.axis("off")
plt.title("draw contours")
plt.subplot(1,3,3)
plt.imshow(result,cmap="gray")

plt.axis("off")
plt.title("Detected sudoko Image")
plt.show()
  
cv2.waitKey()
cv2.destroyAllWindows()