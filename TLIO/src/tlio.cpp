#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>


#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <livox_ros_driver2/CustomMsg.h>


#include <condition_variable>


#include "pcl_process.h"
using namespace std;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

mutex mtx_buffer;
condition_variable sig_buffer;


double time_offset_lidar_to_imu = 0.0;//lidar相对于imu时间偏移，配置参数




std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());

bool time_sync_en = false;
bool timediff_set_flg = false;
double timediff_lidar_to_imu = 0.0;
double last_timestamp_lidar = 0.0;
double last_timestamp_imu = -1.0;

std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<double> time_buffer;





void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();

    //如果当前数据包的时间戳小于上一个数据包的时间戳，说明数据包出现了回环，清空缓存
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    //如果时间同步未启用且 IMU 和 LiDAR 的时间差超过 10 秒，并且 IMU 和 LiDAR 缓冲区都不为空，函数会输出一个警告信息，提示 IMU 和 LiDAR 未同步。
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }
    //如果时间同步已启用且时间差标志未设置，并且 IMU 和 LiDAR 的时间差超过 1 秒且 IMU 缓冲区不为空，函数会设置时间差标志，并计算 LiDAR 相对于 IMU 的时间差，然后输出同步信息。
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_to_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_to_imu);
    }
    //将 LiDAR 数据包转换为 PCL 格式，并将其存储到 LiDAR 缓冲区中。
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    pclProcessor->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;

    //将IMU时间戳对齐到LiDAR时间戳
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    if (abs(timediff_lidar_to_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_to_imu + msg_in->header.stamp.toSec());
    }
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_offset_lidar_to_imu);
    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


struct MeasureGroup     // Lidar data and imu dates for the current process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};



int main(int argc, char** argv) {
    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;

    ros::Subscriber sub_pcl = nh.subscribe("/livox/lidar", 200000, livox_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 200000, imu_cbk);

    ros::spin();
    return 0;
}