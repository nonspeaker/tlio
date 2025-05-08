#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_broadcaster.h>
#include <ieskf/use-ikfom.hpp>
class MessagePublisher {
public:
    MessagePublisher(ros::NodeHandle &nh);
    ~MessagePublisher();
    void publishOdometry(const state_ikfom& state, const Eigen::Matrix<double, 24, 24> & P, double timestamp);
    void publishPointCloudIMU(const PointCloudXYZI::Ptr &cloud, double timestamp);
    void publishPointCloudWorld(const PointCloudXYZI::Ptr &cloud, double timestamp);
    void publishPath(const state_ikfom& state, double timestamp);

    void writeOdometryToFile(); // 添加一个方法用于将内存中的数据写入文件

private:

    nav_msgs::Path path;//lidar移动路径信息的消息
    nav_msgs::Odometry odometry;//建图后里程计的消息

    ros::Publisher pubOdometry;
    ros::Publisher pubPointCloudIMU;
    ros::Publisher pubPointCloudWorld;
    
    ros::Publisher pubPath;

    // 写出里程计信息到文件
    std::ofstream odom_file;
    std::vector<nav_msgs::Odometry> odometryBuffer; // 用于存储里程计信息的容器
};


MessagePublisher::MessagePublisher(ros::NodeHandle &nh) {

    //初始化地图时间戳和帧
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    pubOdometry = nh.advertise<nav_msgs::Odometry>("/odometry", 100000);
    pubPointCloudWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubPointCloudIMU = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);


    // 打开文件并以覆盖模式写入表头
    odom_file.open("/home/tuyanchen/Livox2/TLIO/src/TLIO/evaluate/odometry.csv", std::ios::out);
    if (odom_file.is_open()) {
        odom_file << "%time,field.header.seq,field.header.stamp,"
                  << "field.pose.pose.position.x,field.pose.pose.position.y,field.pose.pose.position.z,"
                  << "field.pose.pose.orientation.x,field.pose.pose.orientation.y,field.pose.pose.orientation.z,field.pose.pose.orientation.w,"
                  << "field.twist.twist.linear.x,field.twist.twist.linear.y,field.twist.twist.linear.z,"
                  << "field.twist.twist.angular.x,field.twist.twist.angular.y,field.twist.twist.angular.z\n";

    }
    odometryBuffer.clear(); // 初始化缓冲区

}


MessagePublisher::~MessagePublisher() {

}

void MessagePublisher::publishOdometry(const state_ikfom& state, const Eigen::Matrix<double, 24, 24> & P, double timestamp)
{
    static uint32_t seq = 0; // 静态变量，序列号从 0 开始

    odometry.header.frame_id = "camera_init";
    odometry.child_frame_id = "body";
    odometry.header.stamp = ros::Time().fromSec(timestamp); // ros::Time().fromSec(lidar_end_time);
    odometry.header.seq = seq++; // 每次调用递增序列号

    odometry.pose.pose.position.x = state.pos(0);
    odometry.pose.pose.position.y = state.pos(1);
    odometry.pose.pose.position.z = state.pos(2);

    auto q_ = Eigen::Quaterniond(state.rot.matrix());
    odometry.pose.pose.orientation.x = q_.coeffs()[0];
    odometry.pose.pose.orientation.y = q_.coeffs()[1];
    odometry.pose.pose.orientation.z = q_.coeffs()[2];
    odometry.pose.pose.orientation.w = q_.coeffs()[3];

    pubOdometry.publish(odometry);

    odometryBuffer.push_back(odometry);

    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odometry.pose.covariance[i * 6 + 0] = P(k, 3);
        odometry.pose.covariance[i * 6 + 1] = P(k, 4);
        odometry.pose.covariance[i * 6 + 2] = P(k, 5);
        odometry.pose.covariance[i * 6 + 3] = P(k, 0);
        odometry.pose.covariance[i * 6 + 4] = P(k, 1);
        odometry.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z));
    q.setW(odometry.pose.pose.orientation.w);
    q.setX(odometry.pose.pose.orientation.x);
    q.setY(odometry.pose.pose.orientation.y);
    q.setZ(odometry.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odometry.header.stamp, "camera_init", "body"));

}

void MessagePublisher::publishPointCloudWorld(const PointCloudXYZI::Ptr &cloud, double timestamp) {
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.stamp = ros::Time().fromSec(timestamp);
    cloudMsg.header.frame_id = "camera_init";
    pubPointCloudWorld.publish(cloudMsg);
}
void MessagePublisher::publishPointCloudIMU(const PointCloudXYZI::Ptr &cloud, double timestamp) {
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.stamp = ros::Time().fromSec(timestamp);
    cloudMsg.header.frame_id = "body";
    pubPointCloudIMU.publish(cloudMsg);
}
void MessagePublisher::publishPath(const state_ikfom& state, double timestamp) {

    geometry_msgs::PoseStamped poseStamped;

    poseStamped.pose.position.x = state.pos(0);
    poseStamped.pose.position.y = state.pos(1);
    poseStamped.pose.position.z = state.pos(2);

    auto q_ = Eigen::Quaterniond(state.rot.matrix());
    poseStamped.pose.orientation.x = q_.coeffs()[0];
    poseStamped.pose.orientation.y = q_.coeffs()[1];
    poseStamped.pose.orientation.z = q_.coeffs()[2];
    poseStamped.pose.orientation.w = q_.coeffs()[3];


    poseStamped.header.stamp = ros::Time().fromSec(timestamp);
    poseStamped.header.frame_id = "camera_init";

    static int counter = 0;
    counter++;
    if (counter % 10 == 0) { // 控制发布频率
        path.poses.push_back(poseStamped);
        pubPath.publish(path);
    }
}

void MessagePublisher::writeOdometryToFile() {
    if (odom_file.is_open()) {
        for (const auto& odometry : odometryBuffer) {
            odom_file << std::scientific << std::setprecision(14)
                      << odometry.header.stamp.toNSec() << "," // 时间戳（纳秒）
                      << odometry.header.seq << ","           // 序列号
                      << odometry.header.stamp.toNSec() << ","; // 时间戳（纳秒）

            odom_file << std::setprecision(15)
                      << odometry.pose.pose.position.x << ","
                      << odometry.pose.pose.position.y << ","
                      << odometry.pose.pose.position.z << "," // 位置
                      << odometry.pose.pose.orientation.x << ","
                      << odometry.pose.pose.orientation.y << ","
                      << odometry.pose.pose.orientation.z << ","
                      << odometry.pose.pose.orientation.w << "\n"; // 旋转（四元数）
        }
        odom_file.flush(); // 刷新缓冲区，确保数据写入磁盘
        odom_file.close(); // 关闭文件
        ROS_INFO("Odometry data successfully written to file: /home/tuyanchen/Livox2/TLIO/src/TLIO/evaluate/odometry.csv");
    } else {
        ROS_WARN("Failed to write odometry data: file is not open.");
    }
}