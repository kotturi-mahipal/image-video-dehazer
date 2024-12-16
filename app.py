from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import math
import numpy as np
from PIL import Image
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T

app = Flask(__name__)

# Image Dehazing
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value=255, high_value=255):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)
    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)
    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)
    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        assert len(flat.shape) == 1
        flat = np.sort(flat)
        n_cols = flat.shape[0]
        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
        thresholded = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)
    return cv2.merge(out_channels)

# Video Dehazing
def video_dehaze(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret == True):
            frame = cv2.resize(frame, (frame_width, frame_height))
            out = simplest_cb(frame, 1)
            out = cv2.flip(out, 0)
            save.write(out)
            cv2.imshow('Dehazed Video', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    save.release()
    cv2.destroyAllWindows()

# Object Detection using YOLO
def detect_objects(frame, net, classes, output_layers, confidence_threshold=0.5, nms_threshold=0.4):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dehaze_image', methods=['POST'])
def dehaze_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    image_path = os.path.join('static','uploads', file.filename)
    file.save(image_path)
    image = cv2.imread(image_path)
    out_image = simplest_cb(image, 1)
    cv2.imwrite(image_path, out_image)
    return redirect(url_for('get_image', filename=file.filename))

@app.route('/dehaze_video', methods=['POST'])
def dehaze_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)
    output_video_path = os.path.join('static', 'output', 'dehazed_' + file.filename)
    video_dehaze(video_path, output_video_path)
    return redirect(url_for('get_video', filename='dehazed_' + file.filename))

@app.route('/get_image/<filename>')
def get_image(filename):
    return render_template('show_image.html', image_name=filename)

@app.route('/get_video/<filename>')
def get_video(filename):
    return render_template('show_video.html', video_name=filename)

if __name__ == '__main__':
    app.run(debug=True)