import cv2
import os
'''
[0:196] landmark
[196:200] bbox
[200:205] attributes
[205] file name
transfer into:
[0] file name
[1:197] landmark
[197:201] bbox
'''
trainf = open('./WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt', 'r')
testf = open('./WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt', 'r')

def createDir(parentdir, dirname):
    if dirname not in os.listdir(parentdir):
        os.mkdir(os.path.join(parentdir, dirname))


# intercept broader area containing human faces for training
def boundarea(img, bbox, landmark):
    x,y,a,b = bbox
    w = a-x
    h = b-y
    center_x = (x+a)//2
    center_y = (y+b)//2
    w = int(w*1.25)
    h = int(h*1.25)
    x_new = center_x - w//2
    if x_new<0:
        x_new = 0
    y_new = center_y - h//2
    if y_new<0:
        y_new = 0
    new_bbox = [x-x_new, y-y_new, a-x_new, b-y_new]
    for ind in range(0, len(landmark), 2):
        landmark[ind] = landmark[ind] - x_new
        landmark[ind+1] = landmark[ind+1] - y_new
    return img[y_new:y_new+h, x_new:x_new+w, :], new_bbox, landmark
count = 0
pose = 0
expression = 0
illumi = 0
make_up = 0
occlusion = 0
blur = 0
for line in trainf.readlines():
    line = line.strip().split()
    landmark = line[0:196]
    bbox = line[196:200]
    attributes = line[200:206]
    name = line[-1]
    img = cv2.imread('./WFLW/WFLW_images/'+name)
    if img is None:
        raise RuntimeError('Unable to open image')
    h, w = img.shape[0:2]
    landmark = list(map(float, landmark))
    bbox = list(map(int, bbox))
    attributes = list(map(int, attributes))
    if attributes[0] == 1:
        pose+=1
    if attributes[1] == 1:
        expression+=1
    if attributes[2]==1:
        illumi+=1
    if attributes[3]==1:
        make_up+=1
    if attributes[4]==1:
        occlusion+=1
    if attributes[5]==1:
        blur+=1
    landmark_x = []
    landmark_y = []
    for i in range(0, 196):
        if (i+1)%2==0:
            landmark_y.append(landmark[i])
        else:
            landmark_x.append(landmark[i])
    xmax = max(landmark_x)
    xmin = min(landmark_x)
    ymax = max(landmark_y)
    ymin = min(landmark_y)
    xmax = xmax if xmax<w else w-1
    xmin = xmin if xmin>0 else 0
    ymax = ymax if ymax<h else h-1
    ymin = ymin if ymin>0 else 0
    if xmax > bbox[2]:
        bbox[2] = int(xmax)
    if xmin < bbox[0]:
        bbox[0] = int(xmin)
    if ymax > bbox[3]:
        bbox[3] = int(ymax)
    if ymin < bbox[1]:
        bbox[1] = int(ymin)
    cropped_rect, bbox_rev, landmark_rev = boundarea(img, bbox, landmark)
    createDir('./', 'Img Dataset')
    createDir('./Img Dataset', 'imgs')
    createDir('./Img Dataset', 'annotations')
    createDir('./Img Dataset/imgs', 'train')
    file_name = "./Img Dataset/imgs/train/{}".format(count)+'_'+name.split('/')[1]
    cv2.imwrite(file_name, cropped_rect)
    newtrainf = open('./Img Dataset/annotations/train_annotation.txt', 'a')
    if count==0:
        newtrainf.truncate(0)
    anno = file_name.split('/')[-1]+' '
    for ind in range(0, len(landmark_rev)):
        anno = anno + str(landmark_rev[ind]) + ' '
    for bx in bbox_rev:
        anno = anno + str(bx) + ' '
    for attribute in attributes:
        anno = anno + str(attribute) + ' '
    anno = anno + '\n'
    newtrainf.write(anno)
    count += 1
    print('train file {} is saved'.format(name.split('/')[1]))
newtrainf.write('{}\n'.format(pose))  # pose number
newtrainf.write('{}\n'.format(expression))  # expression number
newtrainf.write('{}\n'.format(illumi))  # illumination number
newtrainf.write('{}\n'.format(make_up))  # make up number
newtrainf.write('{}\n'.format(occlusion))  # occlusion number
newtrainf.write('{}\n'.format(blur))  # blur number
newtrainf.write('{}\n'.format(count))  # total number
newtrainf.close()
trainf.close()

test_count = 0
for line in testf.readlines():
    line = line.strip().split()
    landmark = line[0:196]
    bbox = line[196:200]
    attributes = line[200:206]
    name = line[-1]
    img = cv2.imread('./WFLW/WFLW_images/'+name)
    if img is None:
        raise RuntimeError('Unable to open image')
    h, w = img.shape[0:2]
    landmark = list(map(float, landmark))
    bbox = list(map(int, bbox))
    attributes = list(map(int, attributes))
    landmark_x = []
    landmark_y = []
    for i in range(0, 196):
        if (i+1)%2==0:
            landmark_y.append(landmark[i])
        else:
            landmark_x.append(landmark[i])
    xmax = max(landmark_x)
    xmin = min(landmark_x)
    ymax = max(landmark_y)
    ymin = min(landmark_y)
    xmax = xmax if xmax<w else w-1
    xmin = xmin if xmin>0 else 0
    ymax = ymax if ymax<h else h-1
    ymin = ymin if ymin>0 else 0
    if xmax > bbox[2]:
        bbox[2] = int(xmax)
    if xmin < bbox[0]:
        bbox[0] = int(xmin)
    if ymax > bbox[3]:
        bbox[3] = int(ymax)
    if ymin < bbox[1]:
        bbox[1] = int(ymin)
    cropped_rect, bbox_rev, landmark_rev = boundarea(img, bbox, landmark)
    createDir('./Img Dataset/imgs', 'test')
    file_name = "./Img Dataset/imgs/test/{}".format(count)+'_'+name.split('/')[1]
    cv2.imwrite(file_name, cropped_rect)
    newtestf = open('./Img Dataset/annotations/test_annotation.txt', 'a')
    if test_count==0:
        newtestf.truncate(0)
    anno = file_name.split('/')[-1]+' '
    for ind in range(0, len(landmark_rev)):
        anno = anno + str(landmark_rev[ind]) + ' '
    for bx in bbox_rev:
        anno = anno + str(bx) + ' '
    for attribute in attributes:
        anno = anno + str(attribute) + ' '
    anno = anno + '\n'
    newtestf.write(anno)
    test_count += 1
    print('test file {} is saved'.format(file_name.split('/')[-1]))
newtestf.close()
