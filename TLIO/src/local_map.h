#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H

#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <ikd-Tree/ikd_Tree.h>
#include "use-ikfom.hpp"
#include "common_lib.hpp"

using namespace std;
using namespace Eigen;

class LocalMapManager
{ 
public:
    LocalMapManager();
    ~LocalMapManager();

    void setParams(float detRange, double filterSizeMapMin, float cubeLen);

    void transformToWorld(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state);

    void updateLocalMapRange(const Eigen::Vector3d &lidarPosition, KD_TREE &ikdtree);
    void updateMapIncremental(const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, KD_TREE &ikdtree, 
                              const vector<PointVector> &nearestPoints, const state_ikfom &state, bool &is_ekf_init);
    void initializeKdTree(KD_TREE &ikdtree, const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, 
                          const state_ikfom &state);
    void savePointCloud(const pcl::PointCloud<PointType>::Ptr& cloud, const std::string& filename);

private:


    //k-dtree
    float mov_threshold;

    float det_range;
    float cube_len;
    double filter_size_map_min;


    bool isLocalMapInit = false;
    BoxPointType localmapRange;
    vector<BoxPointType> cubNeedRm;

    void pointLidarToWorld(const PointType &pi, PointType &po, const state_ikfom &state);
};

#endif 