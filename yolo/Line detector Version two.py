import numpy as np
import cv2
import time

#------------------------------------------------------------------------------------------------

def Find_Edge(image_send):
    try:
        image = cv2.resize(image_send, (640, 320), interpolation=cv2.INTER_AREA)        
        
        for value in range(20, 255, 3):
            
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
                
        edges = cv2.Canny(contour_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(contour_img, 1, np.pi/180, 60)
    
        rho, theta = lines[0][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1,y1), (x2,y2), (255,255,0), 2, cv2.LINE_AA)
        
        return image, (x1, y1), (x2, y2)
    
    except:
        return image, (0, 0), (0, 0)

#------------------------------------------------------------------------------------------------

def draw_dashed_line(img, x1, y1, x2, y2, color=(0, 0, 255), thickness=1, dash_len=10, gap_len=5):
    height, width, _ = img.shape
    dx, dy = x2 - x1, y2 - y1
    line_len = int(np.sqrt(dx**2 + dy**2))
    theta = np.arctan2(dy, dx)

    dash_dx = int(np.cos(theta) * dash_len)
    dash_dy = int(np.sin(theta) * dash_len)
    gap_dx = int(np.cos(theta) * gap_len)
    gap_dy = int(np.sin(theta) * gap_len)
    px, py = x1, y1

    for i in range(line_len // (dash_len + gap_len)):
        cv2.line(img, (px, py), (px + dash_dx, py + dash_dy), color, thickness)
        px += (dash_dx + gap_dx)
        py += (dash_dy + gap_dy)

#------------------------------------------------------------------------------------------------
        
def get_intersection_point(line1, line2):
    # Extract endpoints of the two lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate the slopes of the two lines
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else None
    slope2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else None
    
    # Check if the two lines are parallel or perpendicular
    if slope1 is None and slope2 is None:
        return None  # Two vertical lines, no intersection
    elif slope1 is None:
        X = x1
        Y = slope2 * (X - x3) + y3
    elif slope2 is None:
        X = x3
        Y = slope1 * (X - x1) + y1
    elif slope1 == 0 and slope2 == 0:
        return None  # Two horizontal lines, no intersection
    elif slope1 == 0:
        Y = y1
        X = (Y - y3) / slope2 + x3
    elif slope2 == 0:
        Y = y3
        X = (Y - y1) / slope1 + x1
    else:
        X = (y3 - y1 + slope1*x1 - slope2*x3) / (slope1 - slope2)
        Y = slope1 * (X - x1) + y1
        
    # Return the intersection point
    return int(X), int(Y)

#------------------------------------------------------------------------------------------------

cap = cv2.VideoCapture('A:/Internship MAS/03.03.2023/edge detector dataset/Video1.mp4')

while True:
    ret, frame = cap.read()
    return_image, coordinate_1, coordinate_2 = Find_Edge(frame)

    height, width, _ = return_image.shape
    draw_dashed_line(return_image, width//2, 0, width//2, height + 20, (0, 0, 0), 1, 10, cv2.LINE_AA)
    draw_dashed_line(return_image, 0, height//5, width+20, height//5, (0, 0, 0), 1, 10, cv2.LINE_AA)
    draw_dashed_line(return_image, 0, (height//5)*4, width+20, (height//5)*4, (0, 0, 0), 1, 10, cv2.LINE_AA)
    
    intersection_point_1 = get_intersection_point((coordinate_1 + coordinate_2), (0, height//5, width, height//5))
    cv2.circle(return_image, intersection_point_1, 10, (0, 0, 255), thickness=2)
    
    intersection_point_2 = get_intersection_point((coordinate_1 + coordinate_2), (0, (height//5)*4, width+20, (height//5)*4))
    cv2.circle(return_image, intersection_point_2, 10, (255, 0, 0), thickness=2)
    
    cv2.line(return_image, intersection_point_1, (width//2, height//5), (0,0,255), 1, cv2.LINE_AA)
    cv2.line(return_image, intersection_point_2, (width//2, (height//5) * 4), (255,0,0), 1, cv2.LINE_AA)
    
    space = np.zeros((50, width, 3), dtype=np.uint8)
    cv2.putText(space, "X = "+str(intersection_point_1[0]-(width//2)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(space, "X = "+str(intersection_point_2[0]-(width//2)), (440, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_AREA)
    image = return_image
    
    merged_img = cv2.vconcat([image, space, frame])
    
    cv2.imshow('image', merged_img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()