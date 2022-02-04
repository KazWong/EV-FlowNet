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

from model import *
from vis_utils import *

class event_image:
    def __init__(self): #window_size is the duration to keep and slice to cut
        self.container = None
        self.count_img = None
        self.time_img = None
        self.lock = threading.Lock()
        self.r_t = np.array([-100.0])
        self.iter_list = np.array([0])

        #th = threading.Thread(target=self.AutoClean)
        #th.start()
    
    def __del__(self):
        #th.join()
        return
    
    def Config(self, win, config):
        self.window_size = win
        self.iter = 0
        self.event_last_time = (rospy.Time.now()).to_sec()
        self.in_weight = config[0]
        self.in_height = config[1]
        self.out_weight = config[2]
        self.out_height = config[3]
        self.weight_ratio = float(self.out_weight) / float(self.in_weight)
        self.height_ratio = float(self.out_height) / float(self.in_height)
        self.count_imgs = np.zeros((1, self.out_weight, self.out_height, 2))
        self.time_imgs = np.zeros((1, self.out_weight, self.out_height, 2))
    
    def append(self, events):
        count_img = np.zeros((self.out_weight, self.out_height, 2))
        time_img = np.zeros((self.out_weight, self.out_height, 2))
        event_in = False
        start_e_t = events.events[0].ts.to_sec()
        start_r_t = (rospy.Time.now()).to_sec()
        
        if (start_e_t > (self.event_last_time + self.window_size)):
            self.event_last_time = start_e_t
        
        for event in events.events:
            e_t = (event.ts).to_sec()
            y = event.x - 192 #int(math.floor(float(event.x)*self.weight_ratio))
            x = event.y - 112 #int(math.floor(float(event.y)*self.height_ratio))
            idx = int(not event.polarity)
            t = (event.ts).to_sec() - start_e_t - ((self.event_last_time + self.window_size) - start_e_t)
            
            if (e_t > self.event_last_time + self.window_size):
                self.lock.acquire()
                self.r_t = np.append(self.r_t, start_r_t + e_t - start_e_t)
                self.iter_list = np.append(self.iter_list, self.iter)
                self.count_imgs = np.append(self.count_imgs, [count_img], axis=0)
                self.time_imgs = np.append(self.time_imgs, [time_img], axis=0)
                self.lock.release()
                
                self.event_last_time = e_t
                count_img = np.zeros((self.out_weight, self.out_height, 2))
                time_img = np.zeros((self.out_weight, self.out_height, 2))
                event_in = False
            else:
                count_img[x, y, idx] += 1
                time_img[x, y, idx] = t
                event_in = True
            
        if (event_in):
            self.lock.acquire()
            self.r_t = np.append(self.r_t, start_r_t + e_t - start_e_t)
            self.iter_list = np.append(self.iter_list, self.iter)
            self.count_imgs = np.append(self.count_imgs, [count_img], axis=0)
            self.time_imgs = np.append(self.time_imgs, [time_img], axis=0)
            self.lock.release()
            event_in = False
        self.iter += 1
    
    def Clean(self):
        now = rospy.Time.now()
        
        #last = now.to_sec() - self.window_size
        #pop = len(filter(lambda x: x < last, self.r_t)) - 1
        
        pop = len(filter(lambda x: x < self.iter - 1, self.iter_list)) - 1
        
        if (pop < 1):
            print("none: ", now.to_sec(), pop)
            return
        
        if ( len(self.r_t) - pop < 1):
            print("size: ", now.to_sec(), pop, len(self.r_t), np.max(self.r_t), np.min(self.r_t))
            return
            
        self.lock.acquire()
        self.count_imgs = np.delete(self.count_imgs, range(pop), axis=0)
        self.time_imgs = np.delete(self.time_imgs, range(pop), axis=0)
        self.r_t = np.delete(self.r_t, range(pop))
        self.iter_list = np.delete(self.iter_list, range(pop))
        self.lock.release()
        
    def CountImg(self):
        self.lock.acquire()
        count_img = np.sum(self.count_imgs, axis=0)
        self.lock.release()
        
        return count_img
    
    def TimeImg(self):
        self.lock.acquire()
        time_img = np.max(self.time_imgs, axis=0)
        self.lock.release()
        return time_img
    

events = event_image()

def callback(msg):
    global events
    
    events.append(msg)


def realtime():
    global events
    
    events.Config(0.033, [256, 256, 256, 256]) # x:192-447, y:112-367
    
    cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)
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
            events.Clean()
            #event_count_image = tf.cast(events.count_img, tf.float32)
            #event_count_image = tf.reshape(event_count_image, (1, 256, 256, 2))
            #event_count_image = tf.reduce_sum(event_count_image, axis=0)
            
            #event_time_image = tf.cast(events.time_img, tf.float32)
            #event_time_image = tf.reshape(event_time_image, (1, 256, 256, 2))
            #event_time_image = tf.reduce_max(event_time_image, axis=0)
            #event_time_image /= tf.reduce_max(event_time_image)
            
            #event_image = tf.concat([event_count_image, event_time_image], 2)
            #event_image = tf.reshape(event_image, (1, 256, 256, 4))
            #event_image = tf.image.resize_image_with_crop_or_pad(event_image, 256, 256)
            time_image = events.TimeImg()
            count_image = events.CountImg()
            
            #print(len(events.r_t), time_image.shape, count_image.shape)
            #print(time_image)
            
            event_time_image = time_image / np.max(time_image)
            data = np.concatenate([count_image, event_time_image], axis=2)
            data = np.expand_dims(data, axis=0)
            data = np.float32(data)
            
            #with tf.variable_scope('vs') as vs:
            #    flow_dict = model(data, is_training=False, do_batch_norm=True)
            #sess.run(tf.global_variables_initializer())
            #sess.run(tf.local_variables_initializer())
            #saver = tf.train.Saver(var_list=var_name_list)
            #saver.restore(sess, '/media/whwong/f38678ef-a4a3-422b-89e7-34f48e2a4dc7/data/ev-flownet/data/log/saver/ev-flownet/model.ckpt-600023')
        
            try:
                flow_dict_np = sess.run(flow_dict, feed_dict={x: data})
                #flow_dict_np = sess.run(flow_dict)
            except tf.errors.OutOfRangeError:
                print("Error")
                break
        
            pred_flow = np.squeeze(flow_dict_np['flow3'])
            pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            
            event_count_image = np.sum(count_image, axis=-1)
            event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
            event_count_image = np.squeeze(event_count_image)
            event_count_image = np.tile(event_count_image[..., np.newaxis], [1, 1, 3])
            
            event_time_image = np.squeeze(np.amax(time_image, axis=-1))
            event_time_image = (event_time_image * 255 / event_time_image.max()).astype(np.uint8)
            event_time_image = np.tile(event_time_image[..., np.newaxis], [1, 1, 3])
            
            #print(event_count_image.shape, event_time_image.shape, pred_flow_rgb.shape)
            
            cat = np.concatenate([event_count_image, event_time_image, pred_flow_rgb], axis=1)
            #cat = pred_flow_rgb
            cat = cat.astype(np.uint8)
            cv2.imshow('EV-FlowNet Results', cat)
            #rospy.sleep(1.0)
            cv2.waitKey(100)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():        
    rospy.init_node('optical_flow', anonymous=True)
    rospy.Subscriber('/dvs/events', EventArray, callback, queue_size = 1)
    
    th = threading.Thread(target=realtime)
    
    th.start()
    rospy.spin()
    th.join()


if __name__ == "__main__":
    main()
