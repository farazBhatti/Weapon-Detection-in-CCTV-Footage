import cv2
import argparse
from keras.models import load_model

from sys import platform
from torch import IntTensor
from predict_VGG import VGG_inference
import numpy as np
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
opt = parser.parse_args()
print(opt)






save_path = 'result.mp4'

img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img


def load_and_preprocess_frame(frame,source):
    
    path, img, im0s, vid_cap  = LoadImages(source, img_size=img_size, half=half)


#    print('image type = ',type(frame))
#    print('image shape = ',frame.shape)
    frame = np.transpose(frame, (2, 0, 1))
#    print('image type after = ',type(frame))
#    print('image shape after = ',frame.shape)
#    exit()
    
    img = torch.from_numpy(frame).to('cpu')
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img
    
    
    
    
def yolo_inference(img):
    print('image type = ',type(img))
    print('image shape = ',img.shape)
#    exit()
    pred = model(img)[0]

    if opt.half:
        pred = pred.float()

        # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
    if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
#    for i, det in enumerate(pred):  # detections per image
#        p, s, im0 = path, '', im0s
        
#        if det is not None and len(det):
#                # Rescale boxes from img_size to im0 size
#                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    return pred



    


def crop_img(img,dim):

    
    x = IntTensor.item(dim[0])#60
    y = IntTensor.item(dim[1])#20
    h = IntTensor.item(dim[2]) #* 1.1#increase by 10 percent to improve VGG classification
    w = IntTensor.item(dim[3]) #* 1.1
    
    

    croped_img = img[int(y):int(y+w), int(x):int(x+h)]
#    cv2.imshow("cropped", croped_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return croped_img




# Initialize model
model = Darknet(opt.cfg, img_size)
print('Model Initialized')



# Load weights
#attempt_download(weights)
model.load_state_dict(torch.load(weights, map_location='cpu')['model'])

# load the trained model from disk
print("[INFO] loading model...")
model_VGG = load_model("VGG_model/weapon.model")
print("Weights Loaded")


# Get classes 
classes = load_classes(parse_data_cfg(opt.data)['names'])
#colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]










vid_cap = cv2.VideoCapture('test_data/videos/v_0.mp4')

fps = vid_cap.get(cv2.CAP_PROP_FPS)

w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
while(vid_cap.isOpened()):
    ret, frame = vid_cap.read()
    if ret==True:
        print('Frame preprocessing')
        frame = load_and_preprocess_frame(frame)
        print('Yolo Inference')
        pred = yolo_inference(frame)
        
#        croped_image = crop_img(pred,dim)
#        vgg_pred = VGG_inference(croped_image)
#        if vgg_pred == 1:
#            
#            
#        else:
#            
#            
#        


        # write the flipped frame
        vid_writer.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
vid_cap.release()
vid_writer.release()
cv2.destroyAllWindows()




def detect(save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    

    
    
    

    # Initialize model
    model = Darknet(opt.cfg, img_size)
    print('Model Initialized')

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
        
        # load the trained model from disk
    print("[INFO] loading model...")
    model_VGG = load_model("VGG_model/weapon.model")
    print("Weights Loaded")

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()


    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    print(colors)

    
    print('Running Inference')
    
    
#    cap = cv2.VideoCapture('vtest.avi')
#
#    while(cap.isOpened()):
#        ret, frame = cap.read()

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()
#        print('Get Detections')
        


        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s


            save_path = str(Path(out) / Path(p).name)
            
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
#                        label = '%s %.2f' % (classes[int(cls)], conf)
#                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
##                        print('################# ', IntTensor.item(xyxy[3]))
#                        print('################# ' )
                        croped_img = crop_img(im0,xyxy)
                        pred_num = VGG_inference(croped_img,model_VGG)
                        label = '%s %.2f' % (classes[int(cls)], conf)

                        if pred_num == 1:
                            
#                            label = '%s %.2f' % (classes[int(cls)], conf)
#                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                            plot_one_box(xyxy, im0, label=label, color=[255,0,0])
                        else:
#                            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#                            print(int(cls))
                            text = "ONLY YOLO3 DETECTION"
                            #print(colors[int(cls)])
                            plot_one_box(xyxy, im0, label=label, color=[0, 0, 255])#colors[int(cls)])

                            cv2.putText(im0, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                	(0, 255, 0), 2)                            
                            
                        vid_writer.write(im0)
                        cv2.imshow('frame',im0)
                        cv2.waitKey(0)

                        
                        
                        print('Results saved to %s' % os.getcwd() + os.sep + out)
                        print('Done. (%.3fs)' % (time.time() - t0))
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        else:
                            break

                            
    vid_cap.release()

    cv2.destroyAllWindows()



                            
                            


#            print('%sDone. (%.3fs)' % (s, time.time() - t))
#
#            # Stream results
#            if view_img:
#                cv2.imshow(p, im0)
#
#            # Save results (image with detections)
#            if save_img:
#                if dataset.mode == 'images':
#                    cv2.imwrite(save_path, im0)
#                else:
#                    if vid_path != save_path:  # new video
#                        vid_path = save_path
#                        if isinstance(vid_writer, cv2.VideoWriter):
#                            vid_writer.release()  # release previous video writer
#
#                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
#                    vid_writer.write(im0)
#
#    if save_txt or save_img:
#        print('Results saved to %s' % os.getcwd() + os.sep + out)
#        if platform == 'darwin':  # MacOS
#            os.system('open ' + out + ' ' + save_path)
#
#    print('Done. (%.3fs)' % (time.time() - t0))


#if __name__ == '__main__':
#parser = argparse.ArgumentParser()
#parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
#parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
#parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
#parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
#parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
#parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
#parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
#parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
#parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
#parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
#parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
#parser.add_argument('--view-img', action='store_true', help='display results')
#opt = parser.parse_args()
#print(opt)

#    with torch.no_grad():
#        detect()

