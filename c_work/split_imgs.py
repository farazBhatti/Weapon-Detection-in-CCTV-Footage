import cv2
import glob
import os

path = './imgs/'
dirName = './split_imgs'


def divide_image(name,img):
    i = 0
    for r in range(0,img.shape[0],224):
        for c in range(0,img.shape[1],224):
            final_img = img[r:r+224, c:c+224,:]
            h,w,_ = final_img.shape

            if w >= 100:
                
                name_img = dirName + name + '_' + str(i) + '_.png' 
#                cv2.imwrite(f"img{r}_{c}.png",final_img)
                

                # Create target Directory if don't exist
                if not os.path.exists(dirName):
                    os.mkdir(dirName)
                    print("Directory " , dirName ,  " Created ")
    
                cv2.imwrite(name_img,final_img)
                print(name_img)
                i = i + 1
                print(i)

            
    





for img in glob.glob(path+'*.png'):
    
    print('processing : ',img)
    image = cv2.imread(img)
    
    if image is None:
        print("Error Opening Image")
    name = img[6:15] 
    print(name)
    print("##########")
    
    img_list = divide_image(name,image)

    
    