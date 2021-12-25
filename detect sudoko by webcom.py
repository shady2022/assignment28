from functools import wraps
import cv2
from imutils.perspective import four_point_transform as im
import numpy as np

video_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, images = video_cap.read()
    cv2.imshow("Sudoku detection", images)
    if not ret:
        break
  
    rows, cols, _ = images.shape
    
    
    img_gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    
    img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
     
    dilate = cv2.dilate (thresh, None, iterations= 1)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    contours = list(contours[0])
    #contours = im.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    

    sudoku_contour = None
    #contour = 0
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)        
        appprox = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(appprox) == 4:
            sudoku_contour = appprox
            break
            
    if sudoku_contour is None:
        pass

    else:
        src_pts = np.array([sudoku_contour[1], sudoku_contour[0], sudoku_contour[3], sudoku_contour[2]], dtype=np.float32)
        dst_pts = np.array([[0, 0],   [700, 0],  [700, 700], [0, 700]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp = cv2.warpPerspective(images, M, (700, 700))
        
        result = cv2.drawContours(images, [sudoku_contour], -1, (0, 255, 0), 5)
        cv2.imshow("frame", result)
    
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('output_webcam.jpg', warp)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
video_cap.release()   
cv2.destroyAllWindows()