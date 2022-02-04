#!/usr/bin/env python3

import os
import time
import threading
import math

import tensorflow as tf
import numpy as np
import glob
import cv2
import rospy

from dvs_msgs.msg import Event
from dvs_msgs.msg import EventArray
from dvs_msgs.msg import CountTimeImg

from model import *
from vis_utils import *

class event_image:
    def __init__(self): #window_size is the duration to keep and slice to cut
        self.count_img = None
        self.time_img = None
        self.lock = threading.Lock()
    
    def __del__(self):
        return
    
    def Config(self, config):
        self.in_width = config[0]
        self.in_height = config[1]
        self.out_width = config[2]
        self.out_height = config[3]
        self.width_ratio = float(self.out_width) / float(self.in_width)
        self.height_ratio = float(self.out_height) / float(self.in_height)
        
        rospy.Subscriber('/dvs/event_img', CountTimeImg, self.callback)
    
    def callback(self, msg):
        self.lock.acquire()
        self.count_img = np.reshape(msg.count, (self.out_width, self.out_height, 2))
        self.time_img = np.reshape(msg.time, (self.out_width, self.out_height, 2))
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
    
    #event_count = 0
    wait = 1
    events.Config([256, 256, 256, 256]) #dvxplorer x:192-447, y:112-367 | davis x:2-257, y:45-300
    
    cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EV-FlowNet Results', 1152,384)
    cv2.waitKey(1)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    with tf.Session(config=config) as sess:
        global_step = tf.train.get_or_create_global_step()
        
        #var_name_list = tf.global_variables()
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_8/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_9/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_10/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_11/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_12/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_13/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_14/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/decoder/conv2d_15/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/transition/conv2d_4/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/transition/conv2d_5/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/transition/conv2d_6/" not in v.name]
        #var_name_list = [v for v in var_name_list if "vs/vs/transition/conv2d_7/" not in v.name]
        
        x = tf.placeholder(tf.float32, shape=(1, 256, 256, 4))
        with tf.variable_scope('vs') as vs:
            flow_dict = model(x, is_training=False, do_batch_norm=True)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #saver = tf.train.Saver(var_list=var_name_list)
        saver = tf.train.Saver()
        saver.restore(sess, '/media/whwong/f38678ef-a4a3-422b-89e7-34f48e2a4dc7/data/ev-flownet/data/log/saver/ev-flownet/model.ckpt-600023')
        
        rospy.sleep(0.1)
        while not rospy.is_shutdown(): 
            time_image = events.TimeImg()
            count_image = events.CountImg()
            
            if (time_image is None or count_image is None):
                cv2.waitKey(wait)
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
        
            pred_flow = np.squeeze(flow_dict_np['flow3'])
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
            cv2.waitKey(wait)
        cv2.destroyAllWindows()


def main():        
    rospy.init_node('optical_flow', anonymous=True)
    
    th = threading.Thread(target=realtime)
    
    th.start()
    rospy.spin()
    th.join()


if __name__ == "__main__":
    main()
