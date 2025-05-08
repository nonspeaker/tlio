#ifndef IMU_PROCESSOR_H
#define IMU_PROCESSOR_H

#include "common_lib.hpp"
#include <ieskf/esekfom.hpp>

#define MAX_INI_COUNT (10)  //IMU初始化时，需要的IMU数据帧数

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};
class ImuProcessor {
public:
    ImuProcessor();
    ~ImuProcessor();

    void set_params(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
    void process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_out);

private:

    bool is_need_init;                  //是否需要初始化
    bool is_first_frame;                //是否是第一帧
    int init_iter_num;                  //初始化迭代次数
    vector<Pose6D> imu_pose_deque;      //存储imu位姿(反向传播用) 

    V3D Lidar_T_wrt_IMU;                //Lidar相对于IMU的平移外参
    M3D Lidar_R_wrt_IMU;                //Lidar相对于IMU的旋转外参
    V3D cov_acc_scale;          //外部传入的初始加速度协方差
    V3D cov_gyr_scale;          //外部传入的初始角速度协方差

    Eigen::Matrix<double, 12, 12> Q;    //噪声协方差矩阵  对应论文式(8)中的Q
    V3D cov_acc;                //加速度协方差  
    V3D cov_gyr;                //角速度协方差
    V3D cov_bias_acc;           //加速度bias的协方差
    V3D cov_bias_gyr;           //角速度bias的协方差

    sensor_msgs::ImuConstPtr last_imu;  //上一个imu队列中的IMU数据
    V3D last_acc;                       //imu队列上一个加速度
    V3D last_gyr;                       //imu队列上一个角速度
    V3D mean_acc;                       //加速度均值，用于计算方差
    V3D mean_gyr;                       //角速度均值，用于计算方差

    double last_lidar_end_time;         //上一雷达帧结束的时间戳


    void Reset();   //重置参数
    void initImu(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);

    void preintegrateIMU(const MeasureGroup &meas, esekfom::esekf &kf_state);

    void removeDistortionByIMU(PointCloudXYZI &pcl_out, esekfom::esekf &kf_state);



};




#endif