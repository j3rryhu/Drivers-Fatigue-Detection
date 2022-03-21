import cv2
import os

path = 'D:/PycharmProjects/98 feature points drivers fatigue detection/Img Dataset/annotations/train_annotation.txt'
file = open(path)
for line in file.readlines():
    line = line.split()
    file_name = line[0]
    img = cv2.imread(os.path.join('D:/PycharmProjects/98 feature points drivers fatigue detection/Img Dataset/imgs/train', file_name))
    for i in range(1, 197, 2):
        x = round(float(line[i]))
        y = round(float(line[i+1]))
        cv2.circle(img, (x, y), 2, color=(0,255,0))
        cv2.imshow(file_name, img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
