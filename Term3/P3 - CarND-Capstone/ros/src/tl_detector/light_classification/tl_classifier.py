from styx_msgs.msg import TrafficLight

import tensorflow as tf
import ros
import rospkg
import os
import rospy
import cv2
import numpy as np

USE_DNN_CLASSIFIER = False
PRINT_IMAGES = False

def load_detection_graph(frozen_graph_filename):
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="detection")
    return graph

def load_classification_graph(frozen_graph_filename):
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="classification")
    return graph

class TLClassifier(object):
    def __init__(self):
        # load classifier

        rp = rospkg.RosPack()

        # Get detection models path
        # Detection models downloaded in https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
         # detection_model_path = "models/detection_models/ssd_mobilenet/frozen_inference_graph.pb"
        detection_model_path = os.path.join(rp.get_path('tl_detector'), 'light_classification', 'models', 'detection_models', 'ssd_mobilenet', 'frozen_inference_graph.pb')
        rospy.loginfo('Detection model in path {}'.format(detection_model_path))
        self.output_images_path = os.path.join(rp.get_path('tl_detector'), 'light_classification', 'output_images')
        self.iteration = 0
        self.light2str = {0: 'red', 1: 'yellow', 2: 'green', 3: 'none', 4: 'unknown'}

        ### Detection model ####
        # Load graph
        self.detection_graph = load_detection_graph(detection_model_path)

        # Start tensor flow session
        self.detection_session = tf.Session(graph=self.detection_graph)

        # (Input) Image: input tensor
        self.detection_image = self.detection_graph.get_tensor_by_name('detection/image_tensor:0')

        # (Output) Boxes: part of the image where an object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection/detection_boxes:0')

        # (Output) Scores: level of confidence for each object detected
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection/detection_scores:0')

        # (Output) Classes: the class of the object detected (10 = traffic light)
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection/detection_classes:0')

        ### Classification model ###
        # Source: https://github.com/tokyo-drift/traffic_light_classifier
        if USE_DNN_CLASSIFIER == True:
            classification_model_path = os.path.join(rp.get_path('tl_detector'), 'light_classification', 'models', 'classification_models', 'tokyo_drift', 'model_classification.pb')
            rospy.loginfo('Classification model in path {}'.format(classification_model_path))
            self.classification_graph = load_classification_graph(classification_model_path)
            self.classification_session = tf.Session(graph=self.classification_graph)
            self.input_graph = self.classification_graph.get_tensor_by_name('classification/input_1_1:0')
            self.output_graph = self.classification_graph.get_tensor_by_name('classification/output_0:0')
            self.index2msg = {0: TrafficLight.RED, 1: TrafficLight.GREEN, 2: TrafficLight.YELLOW}
            self.index2str = {0: 'red', 1: 'green', 2: 'yellow'}


    def get_traffic_light_b_box(self, image):
        """Determines the traffic light bounding box within an image"""

        pred_boxes, pred_scores, pred_classes = self.detection_session.run([self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.detection_image: np.expand_dims(image, axis=0)})

        pred_boxes = pred_boxes.squeeze()
        pred_classes = pred_classes.squeeze()
        pred_scores = pred_scores.squeeze()

        confidence_thresh = 0.15
        max_confidence = 0
        box_idx = -1
        img_h, img_w = image.shape[:2]

        # Select bounding box with highest level of confidence in traffic light class (10)
        for i in range(pred_boxes.shape[0]):
            box_i = pred_boxes[i]
            score_i = pred_scores[i]
            class_i = pred_classes[i]
            if score_i > confidence_thresh and class_i == 10:
                if score_i > max_confidence:
                    box_idx = i
                    max_confidence = score_i

        if box_idx > -1:
            box = pred_boxes[box_idx]

            # Get box coordinates
            x0 = int(box[1] * img_w) #top left horizontal
            x1 = int(box[3] * img_w) #bottom right horizontal
            y0 = int(box[0] * img_h) #top left vertical
            y1 = int(box[2] * img_h) #bottom right vertical

            # Expand boxes
            x0 = max(x0-5,0)
            x1 = min(x1+5, img_w)
            y0 = max(y0-10, 0)
            y1 = min(y1+10,img_h)

            return x0, x1, y0, y1
        else:
            return None

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if PRINT_IMAGES == True:
            im_name = "original_"+str(self.iteration)+".jpg"
            cv2.imwrite(self.output_images_path+"/"+im_name, image)

        tl_b_box = self.get_traffic_light_b_box(image)

        if tl_b_box == None:
            return TrafficLight.UNKNOWN

        x0, x1, y0, y1 = tl_b_box
        tl_b_box_image = image[y0:y1,x0:x1]

        if USE_DNN_CLASSIFIER == True:
            return self.dnn_classifier(tl_b_box_image)
        else:
            return self.cv_classifier_3(tl_b_box_image)

    def dnn_classifier(self, image):
        """
        Source: https://github.com/tokyo-drift/traffic_light_classifier

        Args:
            Image: cv2.image in BGR
        """

        # Resize image for DNN squeezenet graph
        resized_image = cv2.resize(image, (32, 32))
        with self.classification_session.as_default(), self.classification_graph.as_default():
            softmax_prob = list(self.classification_session.run(tf.nn.softmax(self.output_graph.eval(feed_dict={self.input_graph: [resized_image]}))))
            softmax_ind = softmax_prob.index(max(softmax_prob))


        decision = self.index2msg[softmax_ind]

        if PRINT_IMAGES == True:
            im_name_3 = str(self.index2str[softmax_ind])+"_"+str(self.iteration)+".jpg"
            cv2.imwrite(self.output_images_path+"/"+im_name_3, image)
            self.iteration = self.iteration+1

        return decision

    def cv_classifier_1(self, image):
        """
        Suggested by @bostonbio, autobots team member

        Args:
            Image: cv2.image in BGR
        """

        decision = TrafficLight.UNKNOWN
        outim = image.copy();
        testim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lowBound1 = np.array([0,51,50])
        highBound1 = np.array([10,255,255])
        testim1 = cv2.inRange(testim, lowBound1 , highBound1)

        lowBound2 = np.array([170,51,50])
        highBound2 = np.array([180,255,255])
        testim2 = cv2.inRange(testim, lowBound2 , highBound2)

        # testimcombined = cv2.addWeighted(testim1, 1.0, testim2, 1.0, 0.0)
        testimcombined = testim1
        testimblur = cv2.GaussianBlur(testimcombined,(15,15),0)
        c = cv2.HoughCircles(testimblur,cv2.HOUGH_GRADIENT,0.5, 41, param1=70, param2=30,minRadius=7,maxRadius=150)

        if c is not None:
            decision = TrafficLight.RED

        return decision

    def cv_classifier_2(self, image):
        """
        Used in real test by: https://github.com/priya-dwivedi/CarND-Capstone-Carla

        Args:
            Image: cv2.image in BGR

        """

        decision = TrafficLight.UNKNOWN

        brightness = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,-1]
        hs, ws = np.where(brightness >= (brightness.max()-30))
        hs_mean = hs.mean()
        img_h = image.shape[0]
        if hs_mean / img_h < 0.4:
            decision = TrafficLight.RED
        elif hs_mean / img_h >= 0.55:
            decision = TrafficLight.GREEN
        else:
            decision = TrafficLight.YELLOW

        return decision

    def cv_classifier_3(self, image):
        """
        Args:
            Image: cv2.image in BGR

        """

        red_img = image[:,:,2]
        green_img = image[:,:,1]

        if PRINT_IMAGES == True:
            cv2.imwrite(self.output_images_path+"/"+'red_img.jpg', red_img)
            cv2.imwrite(self.output_images_path+"/"+'green_img.jpg', green_img)

        # counts pixels which are max in each picture color:
        red_area = np.sum(red_img == red_img.max())
        green_area = np.sum(green_img == green_img.max())

        if red_area - green_area >= 15:
            decision = TrafficLight.RED
            # rospy.loginfo("RED {0} {1}".format(red_area, green_area))
        elif green_area - red_area >= 15:
            decision = TrafficLight.GREEN
            # rospy.loginfo("GREEN {0} {1}".format(red_area, green_area))
        else:
            decision = TrafficLight.UNKNOWN
            # print("UNKNOWN",red_area, green_area)

        return decision
