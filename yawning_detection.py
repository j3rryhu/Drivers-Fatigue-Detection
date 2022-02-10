from resnet import ResNet
import cv2
import torch
import math


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


ckpt = torch.load('ckpt.pth.tar')
state = ckpt['state']
model = ResNet()
model.load_state_dict(state)
video_file = '' # fill in video file
cap = cv2.VideoCapture(video_file)
while True:
    ret, image = cap.read()
    h, w, _ = image.shape
    if not ret:
        break
    landmark = model(image)
    torch.squeeze(landmark)
    ''' CODE FOR YAWNING DETECTION '''
    # # Mouth point: [88,89,90,91,92,93,94,95]
    # upper_lip_x = landmark[180]
    # upper_lip_y = landmark[181]
    # lower_lip_x = landmark[188]
    # lower_lip_y = landmark[189]
    # mouth_distance = distance(upper_lip_x, upper_lip_y, lower_lip_x, lower_lip_y)
    # if mouth_distance>40:
    #     cv2.putText(image, 'Fatigue behavior detected!', (10, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0))
    # urel_x = landmark[124]
    # urel_y = landmark[125]
    # lrel_x = landmark[132]
    # lrel_y = landmark[133]
    # ulel_x = landmark[140]
    # ulel_y = landmark[141]
    # llel_x = landmark[148]
    # llel_y = landmark[149]
    # if distance(urel_x, urel_y, lrel_x, lrel_y)<5 and distance(ulel_x, ulel_y, llel_x, llel_y)<5:
    #     cv2.putText(image, 'Fatigue behavior detected!', (10, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
    #                 color=(255, 0, 0))
#     cv2.imshow('video', image)
#     if cv2.waitKey(10) == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()
    '''CODE FOR TEST'''
#     for index in range(0, len(landmark), 2):
#         lm_x = landmark[index]
#         lm_y = landmark[index+1]
#         cv2.circle(image, (lm_x, lm_y), 5, (255,0,0), -1)
#     cv2.imshow('video', image)
#     if cv2.waitKey(10) == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()
