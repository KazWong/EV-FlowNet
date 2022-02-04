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
        self.lock_event = threading.Lock()
        self.lock = threading.Lock()
        self.events = np.array([])
        self.th = threading.Thread(target=self.ImageStream)
    
    def __del__(self):
        self.th.join()
        return
    
    def Config(self, fps, nframe, config):
        self.in_weight = config[0]
        self.in_height = config[1]
        self.out_weight = config[2]
        self.out_height = config[3]
        self.weight_ratio = float(self.out_weight) / float(self.in_weight)
        self.height_ratio = float(self.out_height) / float(self.in_height)
        self.count_imgs = np.zeros((1, self.out_weight, self.out_height, 2))
        self.time_imgs = np.float64(np.zeros((1, self.out_weight, self.out_height, 2)))

        self.fps = np.float64(1.0/np.float64(fps))
        self.nframe = nframe
        self.start_t = rospy.Time.now()
        
        self.th.start()
    
    def append(self, events):
        self.lock_event.acquire()
        self.events = np.append(self.events, events.events)
        self.lock_event.release()
    
    def ImageStream(self):
        while not rospy.is_shutdown():
            if (len(self.events) < 1):
                continue
            
            events = None
            self.lock_event.acquire()
            t0 = (self.events[0].ts).to_sec()
            t_plus = t0 + self.fps
            events = filter(lambda x: ((x.ts).to_sec() < t_plus), self.events)
            if (len(events) > 0 and events is not None):
                self.events = np.delete(self.events, range(len(events)))
            
            events_shape = self.events.shape
            if (events_shape[0] > 0):
                et0 = self.events[0].ts.to_sec()
                et1 = self.events[-1].ts.to_sec()
            self.lock_event.release()
            
            if (len(events) == 0 or events is None):
                continue
            
            if ((events_shape[0] > 0) and events is not None):
                print(t0, t_plus, events_shape, len(events), events[0].ts.to_sec(), events[-1].ts.to_sec(), et0, et1)
            
            count_img = np.zeros((self.out_weight, self.out_height, 2))
            time_img = np.float64(np.zeros((self.out_weight, self.out_height, 2)))
            
            for event in events:
                y = event.x #- 192 #int(math.floor(float(event.x)*self.weight_ratio))
                x = event.y #- 112 #int(math.floor(float(event.y)*self.height_ratio))
                t = event.ts.to_sec() - t0
                idx = int(not event.polarity)
                
                count_img[x, y, idx] += 1
                time_img[x, y, idx] = t;
            time_img_np = np.clip(time_img, a_min=0, a_max=None)
            
            self.lock.acquire()
            self.count_imgs = np.append(self.count_imgs, [count_img], axis=0)
            self.time_imgs = np.append(self.time_imgs, [time_img_np], axis=0)
            self.lock.release()
        
    def CountImg(self):
        img_shape = self.count_imgs.shape
        if (img_shape[0] > self.nframe):
            self.lock.acquire()
            count_img = np.sum(self.count_imgs[:self.nframe], axis=0)
            self.count_imgs = np.delete(self.count_imgs, range(self.nframe), axis=0)
            self.lock.release()
        else:
            count_img = None
        
        return count_img
    
    def TimeImg(self):
        img_shape = self.time_imgs.shape
        if (img_shape[0] > self.nframe):
            self.lock.acquire()
            time_img = np.amax(self.time_imgs[:self.nframe], axis=0)
            self.time_imgs = np.delete(self.time_imgs, range(self.nframe), axis=0)
            self.lock.release()
        else:
            time_img = None
        
        return time_img

events = event_image()

def callback(msg):
    global events
    events.append(msg)


def realtime():
    global events
    
    fps = 10
    nframe = 1
    event_count = 0
    wait = int(1000.0/float(fps))
    events.Config(fps, nframe, [256, 256, 256, 256]) #dvxplorer x:192-447, y:112-367 | davis x:2-257, y:45-300
    #event_images = np.zeros((1, 256, 256, 4))
    
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
            
            #event_count += np.sum(count_image)
            #event_images = np.append(event_images, data, axis=0)

            try:
                flow_dict_np = sess.run(flow_dict, feed_dict={x: data})
            except tf.errors.OutOfRangeError:
                print("Error")
                break
        
            pred_flow = np.squeeze(flow_dict_np['flow3'])
            pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            
            #event_count += np.sum(count_image)
            
            event_count_image = np.subtract(count_image[..., 0], count_image[..., 1])
            event_count_image += np.abs(event_count_image.min())
            event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
            event_count_image = np.squeeze(event_count_image)
            event_count_image = np.tile(event_count_image[..., np.newaxis], [1, 1, 3])
            
            event_time_image = np.squeeze(np.amax(time_image, axis=-1))
            event_time_image = (event_time_image * 255 / event_time_image.max()).astype(np.uint8)
            event_time_image = np.tile(event_time_image[..., np.newaxis], [1, 1, 3])
            
            cat = np.concatenate([event_count_image, event_time_image, pred_flow_rgb], axis=1)
            cat = cat.astype(np.uint8)
            cv2.imshow('EV-FlowNet Results', cat)
            cv2.waitKey(wait)
        cv2.destroyAllWindows()


def main():        
    rospy.init_node('optical_flow', anonymous=True)
    rospy.Subscriber('/dvs/events', EventArray, callback)
    
    th = threading.Thread(target=realtime)
    
    th.start()
    rospy.spin()
    th.join()


if __name__ == "__main__":
    main()
