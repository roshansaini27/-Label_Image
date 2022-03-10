import tensorflow as tf
import os

from PIL import Image
import numpy as np
import cv2
import math

graph_def = tf.compat.v1.GraphDef()
labels = []
models=["ssdlite_mobilenet_v2","faster_rcnn","mask_rcnn_inception_v2"]
i=int(input("model_no.="))
filename = "/home/prashamsa/"+models[i]+"/frozen_inference_graph.pb"
print(filename)
labels_filename = "/home/prashamsa/label.txt"

# Import the TF graph
with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
gf=graph

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        print(op.name)
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)
#print(analyze_inputs_outputs(gf))
# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

imageFile = "/home/prashamsa/images/apple.jpeg"
#image = Image.open(imageFile)
image=cv2.imread(imageFile)

h,w = image.shape[:2]
if i==0 :
   new_size = (300,300)
elif i==1:
     if  h >600 and w<1024 :
        new_size = (h,w)
     else:
        new_size = (700,900)
else:
    if  h >800 and w<1365 :
       new_size = (w,h)
    else:
       new_size = (900,1300)
image= cv2.resize(image, new_size)

print(image.shape)
h_i, w_i = image.shape[:2]

with tf.compat.v1.Session() as sess:
    input_tensor_shape = sess.graph.get_tensor_by_name('image_tensor:0').shape.as_list()
network_input_size = input_tensor_shape[1]


output_layer = ['detection_boxes:0','detection_scores:0','num_detections:0','detection_classes:0']
input_node = 'image_tensor:0'

with tf.compat.v1.Session() as sess:
    try:
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        predictions = sess.run([detection_boxes,detection_scores,num_detections,detection_classes],{input_node: [image] })

        print(predictions)
        print(len(predictions))
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        exit(-1)

bounding_boxes=predictions[0][0]
scores=predictions[1][0]
classes=predictions[3][0]
classes=classes.astype(int)
td=predictions[2] #total_detections
threshold=float(input("dtection_threshold"))
new_scores=[]
for f in scores:
    if f<threshold:
        print("")
    else:
       new_scores.append(f)
num=len(new_scores)
print(num)
i=0
while i <=num-1:
    #boundinng bboxes
    bb=bounding_boxes[i]
    image=cv2.rectangle(image,(int(bb[1]*w_i),int(bb[0]*h_i)),(int(bb[3]*w_i),int(bb[2]*h_i)),(255,0,0),3)
    #label and scores
    category=labels[classes[i]-1]
    print(category)
    image=cv2.putText(image,str(category)+str(new_scores[i]),(int(bb[1]*w_i),int(bb[0]*h_i)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    i=i+1
cv2.imshow('DETECTED_IMAGE',image)
cv2.waitKey(0)
