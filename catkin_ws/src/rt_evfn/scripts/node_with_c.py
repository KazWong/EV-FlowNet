#!/usr/bin/env python3

import os
import time
import threading
import math

import numpy as np
import cv2
import rospy

from dvs_msgs.msg import CountTimeImg
from dvs_msgs.msg import PredFlow

from model import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class event_image:
    def __init__(self): #window_size is the duration to keep and slice to cut
        self.count_img = None
        self.time_img = None
        self.lock = threading.Lock()
    
    def __del__(self):
        return
    
    def Config(self, config):
        self.shape = config
        
        rospy.Subscriber('/dvs/event_img', CountTimeImg, self.callback)
    
    def callback(self, msg):
        self.lock.acquire()
        self.count_img = np.reshape(msg.count, (self.shape[1], self.shape[2], 2))
        self.time_img = np.reshape(msg.time, (self.shape[1], self.shape[2], 2))
        self.lock.release()
        
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

events = event_image()

def realtime():
    global events
    
    delay = 1.0/30.0
    events.Config((1, 256, 256, 4))
    pub = rospy.Publisher('/pred_flow', PredFlow, queue_size=1)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    with tf.Session(config=config) as sess:
        global_step = tf.train.get_or_create_global_step()

        x = tf.placeholder(tf.float32, shape=events.shape)
        with tf.variable_scope('vs'):
            flow_dict = model(x, is_training=False, do_batch_norm=True)
        
        #var_name_list = tf.global_variables()
        #print(var_name_list)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        path = os.getcwd() + '/src/rpg_dvs_ros/rt_evfn/data/log/saver/ev-flownet/model.ckpt-600023'
        saver = tf.train.Saver()
        saver.restore(sess, path)
        
        while not rospy.is_shutdown(): 
            time_image = events.TimeImg()
            count_image = events.CountImg()
            
            if (time_image is None or count_image is None):
                rospy.sleep(delay)
                continue
            
            time_image_max = np.float64(np.amax(time_image))
            time_image = np.float64(time_image)
            if (time_image_max > 0.0):
                event_time_image = time_image / time_image_max
            else:
                event_time_image = time_image
            event_time_image = np.float64(event_time_image)
            count_image = np.float64(count_image)
            data = np.concatenate([count_image, event_time_image], axis=2)
            data = np.expand_dims(data, axis=0)
            data = np.float32(data)

            try:
                flow_dict_np = sess.run(flow_dict, feed_dict={x: data})
            except tf.errors.OutOfRangeError:
                print("Error")
                break
                
            pf = PredFlow()
            pf.flow0 = np.squeeze(flow_dict_np['flow0']).flatten().tolist() #(32, 32, 2)
            pf.flow1 = np.squeeze(flow_dict_np['flow1']).flatten().tolist() #(64, 64, 2)
            pf.flow2 = np.squeeze(flow_dict_np['flow2']).flatten().tolist() #(128, 128, 2)
            pf.flow3 = np.squeeze(flow_dict_np['flow3']).flatten().tolist() #(256, 256, 2)
            pub.publish(pf)
            
            rospy.sleep(delay)


def main():        
    rospy.init_node('optical_flow', anonymous=True)
    
    th = threading.Thread(target=realtime)
    
    th.start()
    rospy.spin()
    th.join()


if __name__ == "__main__":
    main()
