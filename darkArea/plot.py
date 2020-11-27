#!/usr/bin/env python

import rosbag
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ax.set_xlim(30,40)
# ax.set_ylim(-200,-190)
# ax.set_zlim(10,20)

ax.set_xlim(0,45)
ax.set_ylim(0,10)
ax.set_zlim(8,12)

fig1, ax1 = plt.subplots()

ax1.set_xlim(0,45)
ax1.set_ylim(0,10)

bag = rosbag.Bag('entrance4.bag')
bridge = CvBridge()
xs = []
ys = []
zs = []
ts = []

frame_cnt = 0
limit = 5000

result = cv.VideoWriter('filename.avi',cv.VideoWriter_fourcc(*'MJPG'),10, (960,720))

for topic, msg, t in bag.read_messages(topics=['/tello/odom', '/tello/camera/image_raw']):
   print("frame: ", frame_cnt)
   # if frame_cnt > limit: break

   if topic == '/tello/odom':
       xs.append(msg.pose.pose.position.x)
       ys.append(msg.pose.pose.position.y)
       zs.append(msg.pose.pose.position.z)
       ts.append(msg.header.stamp.to_sec())
       frame_cnt = frame_cnt + 1

       # ax.plot3D(xs, ys, zs, 'blue')


   elif topic == '/tello/camera/image_raw' and (frame_cnt % 10) == 0:
       cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
       cv.imshow("cam", cv_image)
       try:
           print cv_image.shape
           result.write(cv_image)
       except:
           print("An exception occurred")

       k = cv.waitKey()

       if k == ord('s'):
           cv.imwrite("img.png", cv_image)

       # ax1.imshow(cv_image)


   # plt.pause(0.01)

result.release()

theta = 20
Xs = np.array(xs)*np.cos(theta) + np.array(ys)*np.sin(theta) + 30
Ys = np.array(ys)*np.cos(theta) - np.array(xs)*np.sin(theta) + 30
Zs = np.array(zs) - 40
Ts = np.array(ts)

ax.plot3D([40, 40, 40, 40, 40], [6.65, 6.65, 7.15, 7.15, 6.65] , [10.5, 11.5, 11.5, 10.5, 10.5], 'k')
ax.scatter(40, 6.86, 11.08, color=(1,0,0), marker="x")
ax.plot3D(50 - 700*normalize(Ys[0:865].reshape(-1, 1) , axis=0).ravel(), 100*normalize(Xs[0:865].reshape(-1, 1) , axis=0).ravel(), Zs[0:865], 'blue')

ax1.plot([40, 40], [6.65, 7.15], 'k')
ax1.plot(50 - 700*normalize(Ys[0:865].reshape(-1, 1) , axis=0).ravel(), 100*normalize(Xs[0:865].reshape(-1, 1) , axis=0).ravel(), 'blue')
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.grid()

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

fig2, ax2 = plt.subplots()
# model = np.poly1d(np.polyfit(Ts[17:93], Zs[17:93], 7))
# print model
# f = interp1d(Ts[17:93], moving_average(Zs[17:93]), kind='cubic')
# Ts_new = np.linspace(Ts[18], Ts[92], num=200, endpoint=True)
# ax2.plot(Ts_new, f(Ts_new))
ax2.plot(Ts[17:93], moving_average(Zs[17:93]))

plt.pause(0.01)
plt.show()

bag.close()
