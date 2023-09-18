import cv2
from PIL import Image
import pytesseract

PATH = "example.png"


img = cv2.imread(PATH)
# blur image
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# edge image
edges = cv2.Canny(dst, 100, 200)


# finding contours
cnts = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# We are using either OpenCV v2.4, v4-beta, or v4-official 
# if the length of the contours tuple supplied by cv2.findContoursis '2'; 
# otherwise, we are using either OpenCV v3, v4-pre, or v4-alpha.
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

# finding car plate
location = None
for cntr in cnts:
    approx = cv2.approxPolyDP(cntr, 10, True)
    if len(approx) == 4:
        location = approx
        break

# crop image from location
x, y, w, h = cv2.boundingRect(location)
img_crop = img[y:y+h, x:x+w]

# get text from crop image
img_car_plate = Image.fromarray(img_crop)
text = pytesseract.image_to_string(img_car_plate)

# clear all necessary spaces and enter
text = text.strip()
print(text)

res = cv2.putText(img, text=text, org=(x, y+h+45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (255,0,0),3)

cv2.imshow("anpr", img)
cv2.waitKey(0)