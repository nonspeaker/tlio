#pragma once

#include <vector>
#include <queue>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
//6D位姿点云结构定义
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D     
    PCL_ADD_INTENSITY;  
    float roll;         
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   
} EIGEN_ALIGN16;                    

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

//pcl格式转为Eigen::Affine3f
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

//pcl格式转为gstam::Pose3
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(PointCloudXYZI::Ptr &cloudIn, PointTypePose &transformIn) 
{
    PointCloudXYZI::Ptr cloudOut(new PointCloudXYZI());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // 直接使用 transformIn 提供的位姿进行变换
    Eigen::Affine3f T_w_b = pcl::getTransformation(
        transformIn.x, transformIn.y, transformIn.z,
        transformIn.roll, transformIn.pitch, transformIn.yaw);

    for (int i = 0; i < cloudSize; ++i) {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = T_w_b(0, 0) * pointFrom.x + T_w_b(0, 1) * pointFrom.y + T_w_b(0, 2) * pointFrom.z + T_w_b(0, 3);
        cloudOut->points[i].y = T_w_b(1, 0) * pointFrom.x + T_w_b(1, 1) * pointFrom.y + T_w_b(1, 2) * pointFrom.z + T_w_b(1, 3);
        cloudOut->points[i].z = T_w_b(2, 0) * pointFrom.x + T_w_b(2, 1) * pointFrom.y + T_w_b(2, 2) * pointFrom.z + T_w_b(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}
