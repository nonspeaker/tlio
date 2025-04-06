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
#include "use-ikfom.hpp"
class MessagePublisher {
public:
    MessagePublisher(ros::NodeHandle &nh);

    void publishPointCloud(const PointCloudXYZI::Ptr &cloud, double timestamp);
    void publishPath(nav_msgs::Path &path, const state_ikfom& state, double timestamp);
    void publishLoopConstraints(const visualization_msgs::MarkerArray &markerArray);

private:
    ros::Publisher pubPointCloudWorld;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;
    ros::Publisher pubPathUpdate;
    ros::Publisher pubLoopConstraintEdge;
};


MessagePublisher::MessagePublisher(ros::NodeHandle &nh) {
    pubPointCloudWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    pubPathUpdate = nh.advertise<nav_msgs::Path>("fast_lio_sam/path_update", 100000);
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);
}

void MessagePublisher::publishPointCloud(const PointCloudXYZI::Ptr &cloud, double timestamp) {
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.stamp = ros::Time().fromSec(timestamp);
    cloudMsg.header.frame_id = "camera_init";
    pubPointCloudWorld.publish(cloudMsg);
}

void MessagePublisher::publishPath(nav_msgs::Path &path, const state_ikfom& state, double timestamp) {

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

void MessagePublisher::publishLoopConstraints(const visualization_msgs::MarkerArray &markerArray) {
    pubLoopConstraintEdge.publish(markerArray);
}