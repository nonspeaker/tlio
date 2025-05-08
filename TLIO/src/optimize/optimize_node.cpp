#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include "params.h"
#include "optimization.h"
#include "loopclosure/common.h"
#include "loopclosure/radius/radius.h"


std::mutex laser_mtx, odom_mtx;
std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;


std::map<int, int> loopIndexContainer;                                     // 闭环索引容器
pcl::PointCloud<PointType>::Ptr keyFramePositions3D(new pcl::PointCloud<PointType>()); // 所有关键帧的3D位置信息
pcl::PointCloud<PointTypePose>::Ptr keyFramePoses6D(new pcl::PointCloud<PointTypePose>()); // 所有关键帧的6D位姿（位置+姿态+时间戳）
std::vector<PointCloudXYZI::Ptr> keyCloudVector;                           // 所有关键帧的点云


void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(laser_mtx);
    laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(odom_mtx);
    odom_buffer.push(msg);
}

bool syncPackages(PointCloudXYZI::Ptr &cloud, PointTypePose &pose){
    if (laser_buffer.empty() || odom_buffer.empty()) 
    {
        return false;
    }

    auto laser_msg = laser_buffer.front();
    auto odom_msg = odom_buffer.front();

    double laser_timestamp = laser_msg->header.stamp.toSec();
    double odom_timestamp = odom_msg->header.stamp.toSec();

    if (std::abs(laser_timestamp - odom_timestamp) < 1e-3) {
        pcl::fromROSMsg(*laser_msg, *cloud);

        // 提取里程计信息
        pose.x = odom_msg->pose.pose.position.x;
        pose.y = odom_msg->pose.pose.position.y;
        pose.z = odom_msg->pose.pose.position.z;

        Eigen::Quaterniond q(
            odom_msg->pose.pose.orientation.w,
            odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y,
            odom_msg->pose.pose.orientation.z);

        // 提取欧拉角
        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0); // yaw, pitch, roll
        pose.yaw = euler[0];
        pose.pitch = euler[1];
        pose.roll = euler[2];

        pose.time = laser_timestamp;

        laser_buffer.pop();
        odom_buffer.pop();
        return true;
    } else if (odom_timestamp < laser_timestamp) {
        odom_buffer.pop();
    } else {
        laser_buffer.pop();
    }

    return false;
}


void publishCorrectedPath(ros::Publisher &pubCorrectPath) {
    // 发布优化后的路径
    nav_msgs::Path correctedPath;
    correctedPath.header.stamp = ros::Time::now();
    correctedPath.header.frame_id = "camera_init";

    for (const auto &pose : keyFramePoses6D->points) {
        geometry_msgs::PoseStamped poseStamped;
        poseStamped.header.stamp = ros::Time(pose.time);
        poseStamped.header.frame_id = "camera_init";
        poseStamped.pose.position.x = pose.x;
        poseStamped.pose.position.y = pose.y;
        poseStamped.pose.position.z = pose.z;

        Eigen::Quaterniond q = Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX());
        poseStamped.pose.orientation.x = q.x();
        poseStamped.pose.orientation.y = q.y();
        poseStamped.pose.orientation.z = q.z();
        poseStamped.pose.orientation.w = q.w();

        correctedPath.poses.push_back(poseStamped);
    }

    pubCorrectPath.publish(correctedPath);
}

void visualizeLoopClosures(ros::Publisher &pubLoopMarkers) {

    if(loopIndexContainer.empty())
        return;
    std::string odometryFrame = "camera_init";

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;

    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) {
        int key_cur = it->first;
        int key_pre = it->second;

        geometry_msgs::Point p;
        p.x = keyFramePoses6D->points[key_cur].x;
        p.y = keyFramePoses6D->points[key_cur].y;
        p.z = keyFramePoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);

        p.x = keyFramePoses6D->points[key_pre].x;
        p.y = keyFramePoses6D->points[key_pre].y;
        p.z = keyFramePoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);   

    pubLoopMarkers.publish(markerArray);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "online_demo");
    ros::NodeHandle nh;

    Config config;
    config.loadParamsFromROS(nh);

    RadiusManager radiusManager(config.historyKeyframeSearchRadius, config.historyKeyframeSearchTimeDiff, config.historyKeyframeSearchNum, 
        keyFramePositions3D, keyFramePoses6D, keyCloudVector);

    OptimizationManager optimizer(config.surroundingkeyframeAddingDistThreshold, config.surroundingkeyframeAddingAngleThreshold, 
        config.historyKeyframeFitnessScore, keyFramePositions3D, keyFramePoses6D, keyCloudVector);

    ros::Publisher pubCorrectPath = nh.advertise<nav_msgs::Path>("/path_correct", 100000);
    ros::Publisher pubCorrectCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
    ros::Publisher pubLoopMarkers = nh.advertise<visualization_msgs::MarkerArray>("loop_closure_constraints", 10);

    // 接收话题
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        "/cloud_registered_body", 10, laserCloudHandler);
    ros::Subscriber subOdom =
        nh.subscribe<nav_msgs::Odometry>("/odometry", 100, OdomHandler);

    bool has_loop_flag = false;
    while (ros::ok()) {
        ros::spinOnce();

        PointCloudXYZI::Ptr current_cloud_body(new PointCloudXYZI());
        PointCloudXYZI::Ptr current_cloud_world(new PointCloudXYZI());

        PointTypePose current_pose;
        PointType current_position;
        if (!syncPackages(current_cloud_body, current_pose)) {
            continue;
        }


        if(optimizer.isKeyFrame(current_pose)) {
            // 添加当前帧
            current_position.x = current_pose.x;
            current_position.y = current_pose.y;
            current_position.z = current_pose.z;
            current_position.intensity = keyFramePositions3D->size();
            keyFramePositions3D->push_back(current_position);
            keyFramePoses6D->push_back(current_pose);
            keyCloudVector.push_back(current_cloud_body);
            
            optimizer.addOdomFactor(current_pose);

            int loopKeyCur = keyFramePoses6D->size() - 1;
            int loopKeyPre = -1;
            PointCloudXYZI::Ptr nearKeyframeCloud(new PointCloudXYZI());

            if(radiusManager.detectLoopAndGetICPClouds(loopKeyCur, loopKeyPre, nearKeyframeCloud)) 
            {
                Eigen::Matrix4f correction;
                current_cloud_world = transformPointCloud(current_cloud_body, current_pose);
                if(optimizer.icpAlign(current_cloud_world, nearKeyframeCloud, correction)) {
            
                    has_loop_flag = true;
                    std::cout << "ICP align clouds success." << std::endl;
                    // 保存闭环节点
                    loopIndexContainer[loopKeyCur] = loopKeyPre;
                    // 添加闭环因子
                    optimizer.addLoopFactor(loopKeyCur, loopKeyPre, correction);
                }
                else {
                    std::cout << "ICP align clouds fail." << std::endl;
                }
            }

            // 执行优化，得到优化后的值
            optimizer.optimize(has_loop_flag);

            // 发布优化后的路径
            publishCorrectedPath(pubCorrectPath);
            // 可视化闭环
            visualizeLoopClosures(pubLoopMarkers);

            // 发布因子图优化后的点云
            PointTypePose &latestPose = keyFramePoses6D->points.back();
            PointCloudXYZI::Ptr correct_cloud_world(new PointCloudXYZI());
            correct_cloud_world = transformPointCloud(current_cloud_body, latestPose);
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(*correct_cloud_world, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCorrectCloud.publish(pub_cloud);


            has_loop_flag = false;
        }
        else
        {
            // 如果不是关键帧，则直接发布当前里程计
            /*
            nav_msgs::Odometry odom;
            odom.header.stamp = ros::Time(current_pose.time);
            odom.header.frame_id = "camera_init";
            odom.child_frame_id = "camera_init";
            odom.pose.pose.position.x = current_pose.x;
            odom.pose.pose.position.y = current_pose.y;
            odom.pose.pose.position.z = current_pose.z;
            Eigen::Quaterniond q = Eigen::AngleAxisd(current_pose.yaw, Eigen::Vector3d::UnitZ()) *
                                   Eigen::AngleAxisd(current_pose.pitch, Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(current_pose.roll, Eigen::Vector3d::UnitX());
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            odom.pose.pose.orientation.w = q.w();
            pubCorrectPath.publish(odom);
            */

            // 如果不是关键帧，则直接发布当前帧的全局点云
            current_cloud_world = transformPointCloud(current_cloud_body, current_pose);
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(*current_cloud_world, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCorrectCloud.publish(pub_cloud);
        }


    }

    return 0;
}