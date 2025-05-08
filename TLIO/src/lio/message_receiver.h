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
        MessageReceiver(ros::NodeHandle &nh, std::string& lidar_topic, std::string& imu_topic, std::shared_ptr<PointCloudProcessor> pclProcessor)
            : pclProcessor(pclProcessor) {

            if(pclProcessor->lidar_type == 1)
                subLidar = nh.subscribe(lidar_topic, 200000, &MessageReceiver::livoxLidarCallback, this);
            else
                subLidar = nh.subscribe(lidar_topic, 200000, &MessageReceiver::standardLidarCallback, this);
            subImu = nh.subscribe(imu_topic, 200000, &MessageReceiver::imuCallback, this);
        }
        void setParameters(bool timeSync, double timeDiff);
        bool syncPackages(MeasureGroup &meas, double &lidarEndTime);
    
    private:
        void standardLidarCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void livoxLidarCallback(const livox_ros_driver2::CustomMsg::ConstPtr &msg);
        void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);
    
        ros::Subscriber subLidar;
        ros::Subscriber subImu;
    
        std::deque<PointCloudXYZI::Ptr> lidarBuffer;
        std::deque<sensor_msgs::Imu::ConstPtr> imuBuffer;
        std::deque<double> timeBuffer;
    
        std::mutex bufferMutex;
        std::condition_variable bufferCondition;
    
        bool timeSyncEnabled = false;
        double externalTimeOffset = -0.1;



        double softwaretimesyncoffset = 0.0;
        double lastTimestampLidar = 0.0;
        double lastTimestampImu = -1.0;
        bool timeDiffSetFlag = false;

        int scan_num = 0;
        double lidarMeanScanTime = 0.0; //帧间隔时间

        bool lidarPushed = false;

        std::shared_ptr<PointCloudProcessor> pclProcessor;
    };

    void MessageReceiver::setParameters(bool timeSync, double timeDiff)
    {
        timeSyncEnabled = timeSync;
        externalTimeOffset = timeDiff;
    }



    void MessageReceiver::standardLidarCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(bufferMutex);
    
        if (msg->header.stamp.toSec() < lastTimestampLidar) {
            ROS_ERROR("Lidar loop back, clearing buffer");
            lidarBuffer.clear();
        }
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        pclProcessor->process(msg, ptr);
        lidarBuffer.push_back(ptr);


        lastTimestampLidar = msg->header.stamp.toSec();
        timeBuffer.push_back(lastTimestampLidar);

        bufferCondition.notify_all();
    }


    void MessageReceiver::livoxLidarCallback(const livox_ros_driver2::CustomMsg::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(bufferMutex);
    
        if (msg->header.stamp.toSec() < lastTimestampLidar) {
            ROS_ERROR("Lidar loop back, clearing buffer");
            lidarBuffer.clear();
        }
        lastTimestampLidar = msg->header.stamp.toSec();


        if (!timeSyncEnabled && abs(lastTimestampImu - lastTimestampLidar) > 10.0 && !imuBuffer.empty() && !lidarBuffer.empty())
        {
            printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", lastTimestampImu, lastTimestampLidar);
        }
    
        if (timeSyncEnabled && !timeDiffSetFlag && abs(lastTimestampLidar - lastTimestampImu) > 1 && !imuBuffer.empty())
        {
            timeDiffSetFlag = true;
            softwaretimesyncoffset = lastTimestampLidar + 0.1 - lastTimestampImu;
            printf("Self sync IMU and LiDAR, time diff is %.10lf \n", softwaretimesyncoffset);
        }

    
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        pclProcessor->process(msg, ptr);
        lidarBuffer.push_back(ptr);
        timeBuffer.push_back(lastTimestampLidar);
    
        bufferCondition.notify_all();
    }
    
    void MessageReceiver::imuCallback(const sensor_msgs::Imu::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(bufferMutex);
    
        sensor_msgs::Imu::Ptr imu_msg(new sensor_msgs::Imu(*msg));

        if (abs(softwaretimesyncoffset) > 0.1 && timeSyncEnabled)
        {
            imu_msg->header.stamp =
                ros::Time().fromSec(softwaretimesyncoffset + msg->header.stamp.toSec());
        }
        imu_msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - externalTimeOffset);

        double timestamp = imu_msg->header.stamp.toSec();

        if (timestamp < lastTimestampImu) {
            ROS_WARN("IMU loop back, clearing buffer");
            imuBuffer.clear();
        }
        lastTimestampImu = timestamp;
        imuBuffer.push_back(imu_msg);
        
        bufferCondition.notify_all();
    }
    
    bool MessageReceiver::syncPackages(MeasureGroup &meas, double &lidarEndTime) {
        //std::unique_lock<std::mutex> lock(bufferMutex);
    
        if (lidarBuffer.empty() || imuBuffer.empty()) {
            return false;
        }
    
        if (!lidarPushed) {
            meas.lidar = lidarBuffer.front();
            meas.lidar_beg_time = timeBuffer.front();
    
            if (meas.lidar->points.size() <= 5) {
                lidarEndTime = meas.lidar_beg_time + lidarMeanScanTime;
                ROS_WARN("Too few input point cloud!\n");
            } 
            else if(meas.lidar->points.back().curvature / double(1000) < 0.5 * lidarMeanScanTime)
            {
                lidarEndTime = meas.lidar_beg_time + lidarMeanScanTime;
            }
            else
            {
                scan_num++;
                lidarEndTime = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidarMeanScanTime += (meas.lidar->points.back().curvature / double(1000) - lidarMeanScanTime) / scan_num;  //注意curvature中存储的是相对第一个点的时间
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