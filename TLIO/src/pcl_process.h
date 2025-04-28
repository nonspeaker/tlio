#ifndef POINT_CLOUD_PROCESSOR_H
#define POINT_CLOUD_PROCESSOR_H

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

enum LIDAR_TYPE
{
  AVIA = 1,
  VELO16,
  OUST64,
  RS32
}; //{1, 2, 3, 4}
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};


namespace ouster_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

namespace velodyne_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(std::uint16_t, ring, ring))


namespace rslidar_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    uint8_t intensity;
    uint16_t ring = 0;
    double timestamp = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace rslidar_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(uint16_t, ring, ring)(double, timestamp, timestamp))




class PointCloudProcessor {
public:
    PointCloudProcessor();
    ~PointCloudProcessor();
    void setParams(int lidarType, int scanLine, int scanRate, int timeUnit, double blindZone, bool featureEnabled, int pointFilterNum);

    void avia_handler(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);


    void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void rs_handler(const sensor_msgs::PointCloud2_<allocator<void>>::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    
    void downsamplePointCloud(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, float leafSize);

    
    int lidar_type; //雷达类型
private:
 
    bool given_offset_time; //时间戳是否有效
    int scan_line; //雷达线数
    int scan_rate;
    int time_unit; //时间单位
    double blind; //盲区大小（0.1m）

    bool feature_enabled; //特征点是否开启
    int point_filter_num; //点云滤波数


    float time_unit_scale;
    
    pcl::VoxelGrid<PointType> voxelFilter; // 用于下采样的体素滤波器
};



#endif // POINT_CLOUD_PROCESSOR_H   