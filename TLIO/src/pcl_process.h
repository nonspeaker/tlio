#ifndef POINT_CLOUD_PROCESSOR_H
#define POINT_CLOUD_PROCESSOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <pcl_conversions/pcl_conversions.h>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class PointCloudProcessor {
public:
    PointCloudProcessor();
    ~PointCloudProcessor();
    void process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

private:
    int N_SCANS; //雷达线数
    float blind; //盲区大小（0.1m）


    int point_filter_num; //点云滤波数

};





#endif // POINT_CLOUD_PROCESSOR_H   