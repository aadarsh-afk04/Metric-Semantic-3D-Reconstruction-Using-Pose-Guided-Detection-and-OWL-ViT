# Quick visual check — run once
import cv2
img = cv2.imread("data/frames/frame_000319.png")
x1,y1,x2,y2 = 570, 70, 1210, 790   # your current MANUAL_PC_BOX
cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)

# Show IO panel crop
iox1 = int(x1 + IO_LEFT  * (x2-x1))
iox2 = int(x1 + IO_RIGHT * (x2-x1))
ioy1 = int(y1 + IO_TOP    * (y2-y1))
ioy2 = int(y1 + IO_BOTTOM * (y2-y1))
cv2.rectangle(img, (iox1,ioy1), (iox2,ioy2), (0,255,255), 3)

cv2.imwrite("results/debug_box.jpg", img)
print("Saved debug_box.jpg — check the yellow IO crop covers all ports")