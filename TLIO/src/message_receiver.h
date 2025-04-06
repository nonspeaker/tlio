#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <deque>
#include <mutex>
#include <condition_variable>
#include "pcl_process.h"


class MessageReceiver {
    public:
        MessageReceiver(ros::NodeHandle &nh, std::shared_ptr<PointCloudProcessor> pclProcessor)
            : pclProcessor(pclProcessor) {
            subLidar = nh.subscribe("/livox/lidar", 200000, &MessageReceiver::lidarCallback, this);
            subImu = nh.subscribe("/livox/imu", 200000, &MessageReceiver::imuCallback, this);
        }
    
        bool syncPackages(MeasureGroup &meas, double &lidarEndTime);
    
    private:
        void lidarCallback(const livox_ros_driver2::CustomMsg::ConstPtr &msg);
        void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);
    
        ros::Subscriber subLidar;
        ros::Subscriber subImu;
    
        std::deque<PointCloudXYZI::Ptr> lidarBuffer;
        std::deque<sensor_msgs::Imu::ConstPtr> imuBuffer;
        std::deque<double> timeBuffer;
    
        std::mutex bufferMutex;
        std::condition_variable bufferCondition;
    
        double lastTimestampLidar = 0.0;
        double lastTimestampImu = -1.0;
        bool timeSyncEnabled = false;
        double timeDiffLidarToImu = 0.0;
        bool timeDiffSetFlag = false;
    
        double lidarMeanScanTime = 0.1; //帧间隔时间
        bool lidarPushed = false;

        std::shared_ptr<PointCloudProcessor> pclProcessor;
    };

    void MessageReceiver::lidarCallback(const livox_ros_driver2::CustomMsg::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(bufferMutex);
    
        if (msg->header.stamp.toSec() < lastTimestampLidar) {
            ROS_ERROR("Lidar loop back, clearing buffer");
            lidarBuffer.clear();
        }
        lastTimestampLidar = msg->header.stamp.toSec();
    
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        pclProcessor->process(msg, ptr);
        lidarBuffer.push_back(ptr);
        timeBuffer.push_back(lastTimestampLidar);
    
        bufferCondition.notify_all();
    }
    
    void MessageReceiver::imuCallback(const sensor_msgs::Imu::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(bufferMutex);
    
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < lastTimestampImu) {
            ROS_WARN("IMU loop back, clearing buffer");
            imuBuffer.clear();
        }
        lastTimestampImu = timestamp;
        imuBuffer.push_back(msg);
        bufferCondition.notify_all();
    }
    
    bool MessageReceiver::syncPackages(MeasureGroup &meas, double &lidarEndTime) {
        std::unique_lock<std::mutex> lock(bufferMutex);
    
        if (lidarBuffer.empty() || imuBuffer.empty()) {
            return false;
        }
    
        if (!lidarPushed) {
            meas.lidar = lidarBuffer.front();
            meas.lidar_beg_time = timeBuffer.front();
    
            if (meas.lidar->points.size() <= 5) {
                lidarEndTime = meas.lidar_beg_time + lidarMeanScanTime;
            } else {
                lidarEndTime = meas.lidar_beg_time + meas.lidar->points.back().curvature / 1000.0;
            }
    
            meas.lidar_end_time = lidarEndTime;
            lidarPushed = true;
        }
    
        if (lastTimestampImu < lidarEndTime) {
            return false;
        }
    
        double imuTime = imuBuffer.front()->header.stamp.toSec();
        meas.imu.clear();
        while (!imuBuffer.empty() && imuTime < lidarEndTime) {
            imuTime = imuBuffer.front()->header.stamp.toSec();
            if (imuTime > lidarEndTime) break;
            meas.imu.push_back(imuBuffer.front());
            imuBuffer.pop_front();
        }
    
        lidarBuffer.pop_front();
        timeBuffer.pop_front();
        lidarPushed = false;
    
        return true;
    }