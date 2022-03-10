import tensorflow as tf
import os
import cv2
import xml.etree.ElementTree as xml
import os
import glob
from PyQt5 import QtCore, QtGui, QtWidgets

graph_def = tf.compat.v1.GraphDef()

labels_filename = "label.txt"

with open(labels_filename, 'rt') as f:
    labels = f.readlines()

models_list=["ssdlite_mobilenet_v2","faster_rcnn","mask_rcnn_inception_v2"]


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(503, 570)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")

        self.Image = QtWidgets.QPushButton(Dialog)
        self.Image.clicked.connect(self.getimage)
        self.Image.setObjectName("Image")
        self.gridLayout.addWidget(self.Image, 2, 0, 1, 1)

        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.5)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout.addWidget(self.doubleSpinBox, 5, 2, 1, 1)

        self.previous_image = QtWidgets.QPushButton(Dialog)
        self.previous_image.clicked.connect(self.previousimage)
        self.previous_image.setObjectName("previous_image")
        self.gridLayout.addWidget(self.previous_image, 4, 0, 1, 1)

        self.detection_threshold = QtWidgets.QLabel(Dialog)
        self.detection_threshold.setObjectName("detection_threshold")
        self.gridLayout.addWidget(self.detection_threshold, 4, 2, 1, 1)

        self.save_annotation = QtWidgets.QPushButton(Dialog)
        self.save_annotation.clicked.connect(self.save_anno)
        self.save_annotation.setObjectName("save_annotation")
        self.gridLayout.addWidget(self.save_annotation, 8, 0, 1, 1)

        self.classes = QtWidgets.QComboBox(Dialog)
        self.classes.setObjectName("classes")
        for label in labels:
            self.classes.addItem(label[:-1])
        self.gridLayout.addWidget(self.classes, 7, 5, 1, 1)
        self.gridLayout.addWidget(self.classes, 7, 2, 1, 1)

        self.next_image = QtWidgets.QPushButton(Dialog)
        self.next_image.clicked.connect(self.nextimage)
        self.next_image.setObjectName("next_image")
        self.gridLayout.addWidget(self.next_image, 5, 0, 1, 1)

        self.images = QtWidgets.QComboBox(Dialog)
        self.images.setObjectName("images")
        self.images.addItem("image list")
        self.gridLayout.addWidget(self.images, 6, 0, 1, 1)

        self.select_image = QtWidgets.QPushButton(Dialog)
        self.select_image.clicked.connect(self.selectimage)
        self.select_image.setObjectName("select_image")
        self.gridLayout.addWidget(self.select_image, 7, 0, 1, 1)

        self.label_filter = QtWidgets.QLabel(Dialog)
        self.label_filter.setObjectName("label_filter")
        self.gridLayout.addWidget(self.label_filter, 6, 2, 1, 1)

        self.open_folder = QtWidgets.QPushButton(Dialog)
        self.open_folder.clicked.connect(self.getfolder)
        self.open_folder.setObjectName("open_folder")
        self.gridLayout.addWidget(self.open_folder, 3, 0, 1, 1)

        self.models = QtWidgets.QComboBox(Dialog)
        self.models.setObjectName("models")
        for model in models_list:
            self.models.addItem(model)
        self.gridLayout.addWidget(self.models, 3, 2, 1, 1)

        self.select_model = QtWidgets.QLabel(Dialog)
        self.select_model.setObjectName("select_model")
        self.gridLayout.addWidget(self.select_model, 2, 2, 1, 1)

        self.detect = QtWidgets.QPushButton(Dialog)
        self.detect.clicked.connect(self.detection)
        self.detect.setObjectName("detect")
        self.gridLayout.addWidget(self.detect, 8, 2, 1, 1)

        self.image = QtWidgets.QLabel(Dialog)
        self.image.setObjectName("image")
        self.gridLayout.addWidget(self.image, 2, 1, 7, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Image.setText(_translate("Dialog", "Image"))
        self.previous_image.setText(_translate("Dialog", "Previous Image"))
        self.detection_threshold.setText(_translate("Dialog", "Detection Threshold"))
        self.save_annotation.setText(_translate("Dialog", "Save Annotation"))
        self.next_image.setText(_translate("Dialog", "Next Image"))
        self.select_image.setText(_translate("Dialog", "Select Image"))
        self.label_filter.setText(_translate("Dialog", "Label Filter"))
        self.open_folder.setText(_translate("Dialog", "Directory"))
        self.select_model.setText(_translate("Dialog", "Select Model"))
        self.detect.setText(_translate("Dialog", "Detect"))
        self.image.setText(_translate("Dialog", ""))

    def getimage(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '.', "Image files (*.jpg *.png)")
        global image_path
        image_path = fname[0]
        self.image.setPixmap(QtGui.QPixmap(fname[0]))

    def getfolder(self):
        dir = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
        for f in glob.glob(dir + "**/*", recursive=True):
            self.images.addItem(f)

    def selectimage(self):
        global idx
        global image_path
        idx = self.images.currentIndex()
        image_path = self.images.currentText()
        self.image.setPixmap(QtGui.QPixmap(image_path))

    def nextimage(self):
        global idx
        global image_path
        self.images.setCurrentIndex(idx+1)
        idx = self.images.currentIndex()
        image_path = self.images.currentText()
        self.image.setPixmap(QtGui.QPixmap(image_path))

    def previousimage(self):
        global idx
        global image_path
        self.images.setCurrentIndex(idx-1)
        idx = self.images.currentIndex()
        image_path = self.images.currentText()
        self.image.setPixmap(QtGui.QPixmap(image_path))

    def detection(self):
        model_name = self.models.currentText()
        print(model_name)
        model_name = model_name+"/frozen_inference_graph.pb"
        print(model_name)

        # Import the TF graph
        with tf.io.gfile.GFile(model_name, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        gf=graph

        with open(labels_filename, 'rt') as lf:
            for l in lf:
                labels.append(l.strip())

        #load the images
        img=cv2.imread(image_path)

        #resize the image
        h,w = img.shape[:2]
        global o_h, o_w
        o_h, o_w = h,w
        if model_name==models_list[0]:
           new_size = (300,300)
        elif model_name==models_list[1]:
             if  h >600 and w<1024 :
                new_size = (h,w)
             else:
                new_size = (700,900)
        else:
            if  h >800 and w<1365 :
               new_size = (w,h)
            else:
               new_size = (900,1300)
        img = cv2.resize(img, new_size)

        h_i, w_i = img.shape[:2]
        global r_h, r_w
        r_h, r_w = h_i, w_i

        with tf.compat.v1.Session() as sess:
            input_tensor_shape = sess.graph.get_tensor_by_name('image_tensor:0').shape.as_list()
        network_input_size = input_tensor_shape[1]

        output_layer = ['detection_boxes:0','detection_scores:0','num_detections:0','detection_classes:0']
        input_node = 'image_tensor:0'

        # make prediction
        with tf.compat.v1.Session() as sess:
            try:
                detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
                num_detections = sess.graph.get_tensor_by_name('num_detections:0')
                detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
                predictions = sess.run([detection_boxes,detection_scores,num_detections,detection_classes],{input_node: [img] })

                # print(predictions)
                # print(len(predictions))
            except KeyError:
                print ("Couldn't find classification output layer: " + output_layer + ".")
                print ("Verify this a model exported from an Object Detection project.")
                exit(-1)

        bounding_boxes=predictions[0][0]
        scores=predictions[1][0]
        classes=predictions[3][0]
        classes=classes.astype(int)
        td=predictions[2] #total_detections
        threshold=0.5
        new_scores=[]
        for f in scores:
            if f>=threshold:
                new_scores.append(f)


        num=len(new_scores)

        global bb_coor
        bb_coor = bounding_boxes
        global pre_class
        pre_class = classes
        global sc
        sc = len(new_scores)

        i=0
        while i <=num-1:
            #boundinng bboxes
            bb=bounding_boxes[i]

            img=cv2.rectangle(img,(int(bb[1]*w_i),int(bb[0]*h_i)),(int(bb[3]*w_i),int(bb[2]*h_i)),(255,0,0),3)
            #label and scores
            category=labels[classes[i]-1]
            print(category)
            img=cv2.putText(img,str(category)+str(new_scores[i]),(int(bb[1]*w_i),int(bb[0]*h_i)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            i=i+1

        img = cv2.resize(img, (w, h))
        if os.path.exists("results"):
            pass
        else:
            os.mkdir("results")

        # if (category+"\n") in labels:
        #     self.classes.setCurrentText(category)

        cv2.imwrite('results/result.jpg',img)
        self.image.setPixmap(QtGui.QPixmap('results/result.jpg'))


    def save_anno(self):
        selected_label = self.classes.currentText()

        with open(labels_filename, 'rt') as lf:
            for l in lf:
                labels.append(l.strip())

        if os.path.exists("annotation"):
            pass
        else:
            os.mkdir("annotation")
        i=0
        while i <=sc-1:
            if (selected_label+"\n")==labels[pre_class[i]-1]:
                #creating xml file
                filename = "annotation/"+str(selected_label)+".xml"
                if os.path.exists(filename):
                    pass
                else:
                    open(filename,"wb")

                coor = bb_coor[i]
                root = xml.Element("\nUsers")
                userelement = xml.Element("\nuser")
                root.append(userelement)
                uid = xml.SubElement(userelement, "\# NOTE: uid")
                uid.text = "1"
                ObjectID = xml.SubElement(userelement, "\nclasses")
                ObjectID.text = labels[pre_class[i]-1]
                Object = xml.SubElement(userelement, "\nObject")
                Object.text = selected_label
                x_min = xml.SubElement(userelement, "\nx_min")
                x_min.text = str(coor[1])
                y_min = xml.SubElement(userelement, "\ny_min")
                y_min.text = str(coor[0])
                x_max = xml.SubElement(userelement, "\nx_max")
                x_max.text = str(coor[3])
                y_max = xml.SubElement(userelement, "\ny_max")
                y_max.text = str(coor[2])
                origionalheight = xml.SubElement(userelement, "\norigional height")
                origionalheight.text = str(o_h)
                origionwidht = xml.SubElement(userelement, "\norigional width")
                origionwidht.text = str(o_w)
                resizedheight = xml.SubElement(userelement, "\nresized height")
                resizedheight.text = str(r_h)
                resizedwidth = xml.SubElement(userelement, "\nresized width")
                resizedwidth.text = str(r_w)
                tree = xml.ElementTree(root)

                with open(filename, "wb") as fh:
                    tree.write(fh)

                print("xml file generated")
                i=i+1
            else:
                print("selected_label is not in image")
                i=i+1

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
