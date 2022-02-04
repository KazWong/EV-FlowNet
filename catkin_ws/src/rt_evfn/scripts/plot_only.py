#!/usr/bin/env python3

import os
import time
import threading
import math

import numpy as np
import glob
import cv2
import rospy

from dvs_msgs.msg import CountTimeImg
from dvs_msgs.msg import PredFlow

class event_image:
    def __init__(self): #window_size is the duration to keep and slice to cut
        self.count_img = None
        self.time_img = None
        self.pred_flow = None
        self.lock = threading.Lock()
        self.p_lock = threading.Lock()
    
    def __del__(self):
        return
    
    def Config(self, config):
        self.width = config[0]
        self.height = config[1]
        
        rospy.Subscriber('/dvs/event_img', CountTimeImg, self.callback)
        rospy.Subscriber('/pred_flow', PredFlow, self.p_callback)
    
    def callback(self, msg):
        self.lock.acquire()
        self.count_img = np.reshape(msg.count, (self.width, self.height, 2))
        self.time_img = np.reshape(msg.time, (self.width, self.height, 2))
        self.lock.release()
    
    def p_callback(self, msg):
        self.p_lock.acquire()
        self.pred_flow = np.reshape(msg.flow3, (self.width, self.height, 2))
        self.p_lock.release()
        
    def CountImg(self):
        self.lock.acquire()
        c_img = self.count_img
        self.lock.release()
        
        return c_img
    
    def TimeImg(self):
        self.lock.acquire()
        t_img = self.time_img
        self.lock.release()
        
        return t_img
    
    def PredFlow(self):
        self.p_lock.acquire()
        flow = self.pred_flow
        self.p_lock.release()
        
        return flow

events = event_image()

def flow_viz_np(flow_x, flow_y):
    import cv2
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def realtime():
    global events
    
    events.Config([256, 256])
    
    cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EV-FlowNet Results', 1152,384)
    cv2.waitKey(1)
    
    while not rospy.is_shutdown(): 
        time_image = events.TimeImg()
        count_image = events.CountImg()
        pred_flow = events.PredFlow()
            
        if (time_image is None or count_image is None or pred_flow is None):
            cv2.waitKey(1)
            continue
            
        time_image_max = np.float64(np.amax(time_image))
        time_image = np.float64(time_image)
        if (time_image_max > 0.0):
            event_time_image = time_image / time_image_max
        else:
            event_time_image = time_image
        event_time_image = np.float64(event_time_image)
        count_image = np.float64(count_image)
            
        pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            
        event_count_image = np.subtract(count_image[..., 0], count_image[..., 1])
        event_count_image += np.abs(event_count_image.min())
        if (event_count_image.max() > 0.0):
            event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
        event_count_image = np.squeeze(event_count_image)
        event_count_image = np.tile(event_count_image[..., np.newaxis], [1, 1, 3])
            
        event_time_image = np.squeeze(np.amax(time_image, axis=-1))
        if (event_time_image.max() > 0.0):
            event_time_image = (event_time_image * 255 / event_time_image.max()).astype(np.uint8)
        event_time_image = np.tile(event_time_image[..., np.newaxis], [1, 1, 3])
            
        cat = np.concatenate([event_count_image, event_time_image, pred_flow_rgb], axis=1)
        cat = cat.astype(np.uint8)
        cv2.imshow('EV-FlowNet Results', cat)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def main():        
    rospy.init_node('optical_flow', anonymous=True)
    
    th = threading.Thread(target=realtime)
    
    th.start()
    rospy.spin()
    th.join()


if __name__ == "__main__":
    main()
