#ifndef POINT_CLOUD_PROCESSOR_H
#define POINT_CLOUD_PROCESSOR_H

#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <ikd-Tree/ikd_Tree.h>
#include "use-ikfom.hpp"
#include "common_lib.hpp"


using namespace std;
using namespace Eigen;

class PointCloudProcessor {
public:
    PointCloudProcessor();
    ~PointCloudProcessor();
    void setParams(int lidarType, int scanLine, double blindZone, float detRange, 
        bool featureEnabled, int pointFilterNum, double filterSizeMapMin, float cubeLen);
    // 点云坐标变换
    void transformToWorld(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state);

    // 点云下采样
    void downsamplePointCloud(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, float leafSize);
    
    //更新局部地图范围
    void updateLocalMapRange(const Eigen::Vector3d &lidarPosition, KD_TREE &ikdtree);

    // 增量更新地图
    void updateMapIncremental(const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, KD_TREE &ikdtree, const vector<PointVector> &nearestPoints, const state_ikfom &state);
    

    // 初始化 k-d 树并存储第一帧点云
    void initializeKdTree(KD_TREE &ikdtree, const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, const state_ikfom &state);

    void process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

 
    

private:
 
    int lidar_type; //雷达类型
    int scan_line; //雷达线数
    double blind; //盲区大小（0.1m）
    float det_range;

    bool feature_enabled; //特征点是否开启
    int point_filter_num; //点云滤波数

    //k-dtree
    double filter_size_map_min;
    float mov_threshold;
    float cube_len;

    pcl::VoxelGrid<PointType> voxelFilter; // 用于下采样的体素滤波器

    bool isLocalMapInit = false;
    BoxPointType localmapRange;
    vector<BoxPointType> cubNeedRm;

    void pointLidarToWorld(const PointType &pi, PointType &po, const state_ikfom &state);
};



#endif // POINT_CLOUD_PROCESSOR_H   