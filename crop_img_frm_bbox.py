import cv2
import os
import numpy as np


txt_folder_name = 'labels'
img_folder_name = 'images'


def get_yolo_coordinates(txt_file):
    
    
#path = '1.txt'

    file = open(txt_file,'r')
#    if file is None:
#        return -1,-1,-1,-1
    coordinate_list = file.read().split(' ') 
    X = coordinate_list[1]
    Y = coordinate_list[2]
    W = coordinate_list[3]
    H = coordinate_list[4]
    return X,Y,W,H

def get_img_coordinates(X_yolo, Y_yolo, W_yolo, H_yolo,img_size):
    
    height = img_size[0]
    width = img_size[1]

    
    A = np.array([[1,1],[1,-1]])
    B = np.array([width * float(X_yolo) * 2,float(W_yolo) * width])
    C = np.array([[1,1],[1,-1]])
    D = np.array([height * float(Y_yolo) * 2,float(H_yolo) * height])
    
    xmax, xmin = np.linalg.solve(A,B)
    ymax, ymin = np.linalg.solve(C,D)
    
    x = int(xmin*0.95)
    y = int(ymin*0.95)
    w = int(xmax*1.05) - int(xmin*0.95)
    h = int(ymax*1.05) - int(ymin*0.95)
    
    return x, y, w, h
    
    
def file_len(path):
    try:
        with open(path) as f:
            for i, l in enumerate(f):
                pass
        return i + 1    
    except OSError as e:
        print("Error Opening text file")#, e.errno)
        return 0 
    

for file_name in os.listdir(img_folder_name):

    #print(file_name)
    txt_file = txt_folder_name + '/'+ file_name.replace('.jpg','.txt')
    #print(txt_file)
    img_path = img_folder_name + '/' + file_name
    #print(img_path)
    img = cv2.imread(img_path)
    if img is not None:

        #print(img.shape)
       
        img_dim = img.shape
        #print('image dim = ',img_dim)
        bbox_num = file_len(txt_file)
        
        if bbox_num != 0: 
            
            for i in range(0,bbox_num):
                #print('processing new files...')
                X_yolo, Y_yolo, W_yolo, H_yolo = get_yolo_coordinates(txt_file)
            
                x, y, w, h = get_img_coordinates(X_yolo, Y_yolo, W_yolo, H_yolo,img_dim)
                
                crop_img = img[y:y+h, x:x+w]
                
                new_name = 'vgg_imgs/' + file_name
                print(new_name)
                
                cv2.imwrite(new_name, crop_img)
#                cv2.imshow("cropped", crop_img)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
        else:
            pass
    else:
        print('Error Opening image')
