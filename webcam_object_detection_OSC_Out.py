import cv2

from yolov7 import YOLOv7

from pythonosc import udp_client

#Setup udp port to send detected data back to TD
upd_ip = "127.0.0.1"
udp_port = 7000
udp_port_2 = 7007

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny_384x640.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer   
    boxes, scores, class_ids = yolov7_detector(frame)

    # # Uncomment to show window
    # combined_img = yolov7_detector.draw_detections(frame)
    # cv2.imshow("Detected Objects", combined_img)

    # # Press key q to stop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    box_coord = []
    objects = []
    for i in range(len(boxes)):
        
        x0, y0, x1, y1 = boxes[i]

        objects.append(class_ids[i])
        objects.append('{:.2f}'.format(scores[i] * 100))           
        box_coord.append(boxes[i])
        
        

    osc = objects
    osc_2 = box_coord 
    # send data as osc to TD
    client = udp_client.SimpleUDPClient(upd_ip, udp_port)
    client.send_message("/scores", str(osc))

    client_2 = udp_client.SimpleUDPClient(upd_ip, udp_port_2)
    client_2.send_message("/box", str(osc_2))        
