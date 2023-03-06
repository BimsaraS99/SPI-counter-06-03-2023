import numpy as np
import cv2
import time
start_time = time.time()


image = cv2.imread('A:/Internship MAS/03.03.2023/edge detector dataset/1.jpg')
image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)

for value in range(20, 255, 1):
    
  blurred = cv2.GaussianBlur(image, (1, 1), 0, 0)
  gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
  b, g, r = cv2.split(blurred)

  ret, thresh = cv2.threshold(r, value, 255, cv2.THRESH_BINARY)
  ret, thresh1 = cv2.threshold(g, value, 255, cv2.THRESH_BINARY)
  ret, thresh2 = cv2.threshold(b, value, 255, cv2.THRESH_BINARY)

  result = cv2.bitwise_or(thresh, cv2.bitwise_and(thresh1, thresh2))

  num_white_pix = cv2.countNonZero(result)
  num_black_pix = result.size - num_white_pix

  try:
    ratio = num_white_pix / num_black_pix
  except:
    ratio = 1000

  if ratio < 190:
    break

result = 255 - result

contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = np.zeros_like(result)

for contour in contours:
    if cv2.contourArea(contour) >= 2:
        cv2.drawContours(contour_img, [contour], 0, (255, 255, 255), -1)

#------------------------------------------------------------------------------------------
for i in range(4, 300, 10):
    edges = cv2.Canny(contour_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(contour_img, 1, np.pi/180, i)
    if len(lines) <= 10:
        print(i)
        break


#for line in lines:
rho, theta = lines[0][0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(image, (x1,y1), (x2,y2), (255,255,255), 2)
#    break

#cv2.imshow("Result1", contour_img)
cv2.imshow('image', image)
print(lines)

#height, width = contour_img.shape[:2]
#piece_height, piece_width = height//10, width

#pieces = []
#for i in range(10):
#    y = i * piece_height
#    piece = contour_img[y:y+piece_height, 0:width]
#    pieces.append(piece)

#image = cv2.resize(pieces[3], (640, 360), interpolation=cv2.INTER_AREA)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

cv2.waitKey(0)