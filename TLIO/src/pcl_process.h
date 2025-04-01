#ifndef POINT_CLOUD_PROCESSOR_H
#define POINT_CLOUD_PROCESSOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <pcl_conversions/pcl_conversions.h>


using namespace std;

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