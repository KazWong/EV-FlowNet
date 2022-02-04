#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <algorithm>

#include<opencv2/opencv.hpp>
#include<iostream>

#include <ros/ros.h>
#include "dvs_msgs/EventArray.h"
#include "dvs_msgs/Event.h"
#include "dvs_msgs/CountTimeImg.h"

using namespace std;
using namespace ros;
using namespace cv;

class EventImg {
	private:
		vector<dvs_msgs::Event> events;
		mutex event_mtx;
		double fps;
		int out_size[2];

		NodeHandle nh;
		Subscriber eventarray_sub;
		Publisher eventimg_pub;

	private:
		void EventArrayCallback(const dvs_msgs::EventArray::ConstPtr& msg) {
			if (msg->events.size() < 1)
				return;

			for (int i=0; i<msg->events.size();) {
				dvs_msgs::CountTimeImg img;
				double t0 = msg->events[i].ts.toSec();
				double t_plus = t0 + fps;
				double t = msg->events[i].ts.toSec();
				bool events_img = false;
				double in_width = double(msg->width) - 96.0; //dvxplorer is 640x480, 256/480=0.5333xxx, 480-96=384/256=1.5
				double in_height = double(msg->height) - 96.0;
				//double in_ratio = in_height/in_width; //keep width
				double in_ratio = in_width/in_height; //keep height
				int array_size = 2*out_size[0]*out_size[1];

				img.width = out_size[0];
				img.height = out_size[1];
				img.count.resize(array_size);
				img.time.resize(array_size);

				//keep width
				//double width_ratio = double(out_size[0])/in_width;
				//double height_ratio = double(out_size[0])*in_ratio/in_height;
				//int width_shift = 0;
				//int height_shift = double(in_height*height_ratio - img.height)/2.0;

				//keep height
				double width_ratio = double(out_size[1])*in_ratio/in_width;
				double height_ratio = double(out_size[1])/in_height;
				int width_shift = double(in_width*width_ratio - img.width)/2.0;
				int height_shift = double(in_height*height_ratio - img.height)/2.0;
				
				do {
					dvs_msgs::Event event = msg->events[i];

					if (event.y < 48 || event.y >= 432 || event.x < 48 || event.x >= 592) {
						i++;
						continue;
					}

					int x = floor(double(event.y - 48.0)*height_ratio) - height_shift;
					int y = floor(double(event.x - 48.0)*width_ratio) - width_shift;
					t = event.ts.toSec();

					if (y < 0 || y >= img.width || x < 0 || x >= img.height) {
						i++;
						continue;
					}

					unsigned int idx = 2*int(x*img.width + y) + int(not event.polarity);

					img.count[idx] += 1.0;
					img.time[idx] = (t - t0 < 0.0)? t:(t-t0);
					events_img = true;

					i++;
				} while(t < t_plus && i<msg->events.size() && ros::ok());

				if (events_img) {
					eventimg_pub.publish(img);
				}
			}
		}

	public:
		EventImg(int _fps, int *_size):nh(), fps(1.0/double(_fps)), out_size{_size[0], _size[1]} //out_width, out_height
		{

			eventarray_sub = nh.subscribe<dvs_msgs::EventArray>("/dvs/events", 1, &EventImg::EventArrayCallback, this);
			eventimg_pub = nh.advertise<dvs_msgs::CountTimeImg>("/dvs/event_img", 1);
		}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "flownet_preprocess");

	int size[4] = {256, 256};
	EventImg event_img(30, size);
	ros::spin();
}
