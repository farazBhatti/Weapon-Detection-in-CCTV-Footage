import cv2
import argparse
from keras.models import load_model

from sys import platform
from torch import IntTensor
from predict_VGG import VGG_inference
import numpy as np
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets_weapon import *
from utils.utils import *


def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def LoadImages_(img0,img_size=416, half=False):


        #img0 = crp_img#cv2.imread(path)  # BGR
#        print('Processing image for weaon detection')
        #assert img0 is not None, 'Image Not Found ' + path
        #print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape = img_size)[0]
#        print('image shape = ',img.shape)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        cv2.imshow('img',img0)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        
        img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
#        print('image shape after Transpose = ',img.shape)
#        cv2.imshow('image in loadImage function',img0)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        


        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
#        return path, img, img0, self.cap
        return img, img0#, self.cap
















def crop_img(img,dim):

    
    x = IntTensor.item(dim[0])#60
    y = IntTensor.item(dim[1])#20
    w = IntTensor.item(dim[2]) #* 1.1#increase by 10 percent to improve VGG classification
    h = IntTensor.item(dim[3]) #* 1.1
    
    

    croped_img = img[int(y):int(h), int(x):int(w)]
#    cv2.imshow("cropped", croped_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return croped_img



def detect(croped_img):#,save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else 416  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = 'output', croped_img, 'weights/best.pt', 'store_true', 'store_true'
    #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' )#if ONNX_EXPORT else opt.device)
#    if os.path.exists(out):
#        shutil.rmtree(out)  # delete output folder
#    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet('cfg/yolov3_weapon.cfg', img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    # load the trained model from disk
#    print("[INFO] loading model...")
#    model_VGG = load_model("VGG_model/weapon.model")
#    print("Weights Loaded")

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

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
#    if webcam:
#        view_img = True
#        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
#        dataset = LoadStreams(source, img_size=img_size, half=half)
#    else:
    save_img = True
#    print('Loading Images using Image lib')
    dataset = LoadImages_(source, img_size=img_size, half=half)
#    print('Size of list',len(dataset))
#    print('shape of Image = ',dataset[0].shape)
#    print('type of Image = ',type(dataset[1]))
#    print('Dataset length',len(dataset))
    # Get classes and colors
    classes = load_classes(parse_data_cfg('data/coco_weapon.data')['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
#    for path, img, im0s, vid_cap in dataset:
    
    
    img, im0s =  dataset
    

#    print(len(dataset))
#    print('Inside loop')
#    print('\n shape of return img : ',img.shape)
#    print('\n type of returned image : ',type(img))
    t = time.time()
#        print('type of img:',type(img))
#        print('shape of Image : ',img.shape)
#        exit()

    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
#        print('type of img after:',type(img))
#        print('shape of Image after: ',img.shape)
#    print('shaoe of image = ',img[0].shape)
#    print('length of image = ',len(img))
#    print('Type of image = ',type(img[0]))
    #exit()
    pred = model(img)[0]
#    print('length of prediction = ',len(pred))
#    print('Shape of prediction = ',pred.shape)
#    exit()

#    if 'store_true':
#        print('@@@@@@@@@@@@')
#        exit()
#        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.5)

    # Apply
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
#        print('prediction length:',len(pred))
#        print('prediciton:',pred)
#        exit()
#            if webcam:  # batch_size >= 1
#                p, s, im0 = path[i], '%g: ' % i, im0s[i]
#            else:
#            p, s, im0 = path, '', im0s
        s, im0 =  '', im0s


        #save_path = str(Path(out) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            print('length of detection:',len(det))
            # Rescale boxes from img_size to im0 size
#                return 1
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#            print('length = ',len(det))
#            print('type = ',type(det))
#            print('Detection list = ',det[0][2])


            return 1,det[0][:4]
        else:
            return 0,None
#
#    for img, im0s ,_ in dataset:
#        print(len(dataset))
#        print('Inside loop')
#        print('\n shape of return img : ',img.shape)
#        print('\n type of returned image : ',type(img))
#        t = time.time()
##        print('type of img:',type(img))
##        print('shape of Image : ',img.shape)
##        exit()
#
#        # Get detections
#        img = torch.from_numpy(img).to(device)
#        if img.ndimension() == 3:
#            img = img.unsqueeze(0)
##        print('type of img after:',type(img))
##        print('shape of Image after: ',img.shape)
#        print('shaoe of image = ',img[0].shape)
#        print('length of image = ',len(img))
#        print('Type of image = ',type(img[0]))
#        exit()
#        pred = model(img)[0]
#
#        if 'store_true':
#            pred = pred.float()
#
#        # Apply NMS
#        pred = non_max_suppression(pred, 0.3, 0.5)
#
#        # Apply
#        if classify:
#            pred = apply_classifier(pred, modelc, img, im0s)
#
#        # Process detections
#        for i, det in enumerate(pred):  # detections per image
##            if webcam:  # batch_size >= 1
##                p, s, im0 = path[i], '%g: ' % i, im0s[i]
##            else:
##            p, s, im0 = path, '', im0s
#            s, im0 =  '', im0s
#
#
#            #save_path = str(Path(out) / Path(p).name)
#            s += '%gx%g ' % img.shape[2:]  # print string
#            if det is not None and len(det):
#                # Rescale boxes from img_size to im0 size
##                return 1
#                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                print('Detection list = ',det[0][2])
#
#
#                return 1,det[0][:4]
##
#                # Print results
#                for c in det[:, -1].unique():
#                    n = (det[:, -1] == c).sum()  # detections per class
#                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string
#
#                # Write results
#                for *xyxy, conf, _, cls in det:
#                    if save_txt:  # Write to file
#                        with open(save_path + '.txt', 'a') as file:
#                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
#
#                    if save_img or view_img:  # Add bbox to image
##                        label = '%s %.2f' % (classes[int(cls)], conf)
##                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
###                        print('################# ', IntTensor.item(xyxy[3]))
###                        print('################# ', xyxy)
#                        croped_img = crop_img(im0,xyxy)
#                        
#                        pred_num = VGG_inference(croped_img,model_VGG)
#                        label = '%s %.2f' % (classes[int(cls)], conf)
#                        
#
#                        if pred_num == 1:
#                            
#                            #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
#                            plot_one_box(xyxy, im0, label=label, color = [0,255,0])
#
#
#                        else:
#                            
#                            text = "ONLY YOLO3 DETECTION"
#                            #print(colors[int(cls)])
#                            plot_one_box(xyxy, im0, label=label, color = [0, 0, 255])#colors[int(cls)])
#
#                            cv2.putText(im0, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                	(0, 255, 0), 2)       
#
#
#
#
#
#                            
#                            
#
#
#            print('%sDone. (%.3fs)' % (s, time.time() - t))
#            
#            cv2.imshow(p, im0)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break            # Stream results
##            if view_img:
##                cv2.imshow(p, im0)
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
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
#    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
#    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
#    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
#    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
#    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
#    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
#    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
#    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
#    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
#    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
#    parser.add_argument('--view-img', action='store_true', help='display results')
#    opt = parser.parse_args()
#    print(opt)
#
#    with torch.no_grad():
#        detect()

