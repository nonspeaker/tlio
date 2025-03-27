#pragma once

#include "common_lib.hpp"
#include "esekfom.hpp"

#define MAX_INI_COUNT (10)  //IMU初始化时，需要的IMU数据帧数

class ImuProcessor {
public:
    ImuProcessor();
    ~ImuProcessor();

    void set_params(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
    void process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_out);

private:

    bool is_need_init;                  //是否需要初始化
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

    void initImu(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
    void undistortPointCloud(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out);
};

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

ImuProcessor::ImuProcessor()
{
    is_need_init = true;    

    Q = process_noise_cov();                    //调用use-ikfom.hpp里面的process_noise_cov初始化噪声协方差
    cov_acc = V3D(0.1, 0.1, 0.1);               //加速度协方差初始化
    cov_gyr = V3D(0.1, 0.1, 0.1);               //角速度协方差初始化
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001); //角速度bias协方差初始化
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001); //加速度bias协方差初始化

    mean_acc = V3D(0, 0, 0);
    mean_gyr = V3D(0, 0, 0);
    last_acc = V3D(0, 0, 0);
    last_gyr = V3D(0, 0, 0);

    Lidar_T_wrt_IMU = V3D(0, 0, 0);                   // lidar到IMU的位置外参初始化
    Lidar_R_wrt_IMU = M3D::Identity();                // lidar到IMU的旋转外参初始化
    last_imu.reset(new sensor_msgs::Imu());           //上一帧imu初始化
}

ImuProcessor::~ImuProcessor() {}

//传入外部参数
void ImuProcessor::set_params(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;//Lidar相对于IMU的平移外参
  Lidar_R_wrt_IMU = rot;   //Lidar相对于IMU的旋转外参
  cov_gyr_scale = gyr;     //外部传入的初始角速度协方差
  cov_acc_scale = acc;     //外部传入的初始加速度协方差
  cov_bias_gyr = gyr_bias; //角速度bias的协方差
  cov_bias_acc = acc_bias; //加速度bias的协方差
}

//初始化IMU：初始化卡尔曼滤波器
 void ImuProcessor::initImu(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
 {
    V3D cur_acc;
    V3D cur_gyr;

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;  //更新加速度均值
        mean_gyr += (cur_gyr - mean_gyr) / N;  //更新角速度均值

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;                //更新加速度方差
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);    //更新角速度方差

        N++;
    }

    //与卡尔曼滤波器的相关初始化
    state_ikfom init_state = kf_state.get_x();
    init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    //重力加速度分量（测量的平均速度的单位方向向量 * 重力加速度预设值)
    init_state.bg = mean_gyr;                                   //角速度偏置（测量的平均角速度）
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;                  //Lidar相对于IMU的平移外参
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);     //Lidar相对于IMU的旋转外参
    kf_state.change_x(init_state);                                 //将初始化后的状态量赋值给卡尔曼滤波器

    Eigen::Matrix<double, 24, 24> init_P = Eigen::MatrixXd::Identity(24, 24);  //初始化协方差矩阵
    init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
    init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
    init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
    init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
    init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
    kf_state.change_P(init_P);  //将初始化后的协方差矩阵赋值给卡尔曼滤波器

    last_imu = meas.imu.back(); //更新上一个IMU数据
}

//消除每个激光雷达点的运动失真（反向传播）
void ImuProcessor::undistortPointCloud(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
    auto imu_deque = meas.imu;   //获取IMU队列
    imu_deque.push_front(last_imu); //将上一帧尾部的IMU数据插入到IMU队列的开头

    const double &imu_end_time = imu_deque.back()->header.stamp.toSec(); //当前帧尾部的imu时间戳
    const double &pcl_beg_time = meas.lidar_beg_time;      //点云开始的时间戳
    const double &pcl_end_time = meas.lidar_end_time;      //点云结束的时间戳

    pcl_out = *(meas.lidar);    //将当前帧的点云赋值给pcl_out
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);//根据每个点的时间戳对点云重排序

    state_ikfom imu_state = kf_state.get_x();  //获取上一次KF估计的后验状态作为本次IMU预测的初始状态
    imu_pose_deque.clear(); //清空IMU位姿队列
    imu_pose_deque.push_back(set_pose6d(0.0, last_acc, last_gyr, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    //将初始状态加入到IMU位姿队列中，包含有时间间隔，上一帧的加速度，角速度，速度，位置，旋转矩阵

    V3D avr_acc; //平均角速度
    V3D avr_gyr; //平均加速度

    double dt = 0;

    input_ikfom in;
    //遍历IMU队列里的每一个IMU数据，计算对应的IMU位姿，放入IMU位姿队列中
    for (auto it_imu = imu_deque.begin(); it_imu < (imu_deque.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);        //拿到当前imu数据
        auto &&tail = *(it_imu + 1);    //拿到下一个imu数据
        //判断时间先后顺序：下一帧imu时间戳是否小于上一帧雷达结束时间戳 不符合直接continue
        if(tail->header.stamp.toSec() < last_lidar_end_time) 
            continue;

        avr_gyr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // 中值积分
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        avr_acc << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        avr_acc  = avr_acc * G_m_s2 / mean_acc.norm(); //通过重力数值对加速度进行调整(除上初始化的IMU大小*9.8)

        //如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
        if(head->header.stamp.toSec() < last_lidar_end_time)
        {
            dt = tail->header.stamp.toSec() - last_lidar_end_time; //从上次雷达时刻末尾开始传播 计算与此次IMU结尾之间的时间差
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();     //两个IMU时刻之间的时间间隔
        }
        in.acc = avr_acc;// 两帧IMU的中值作为输入in  用于前向传播
        in.gyro = avr_gyr;

         // 配置协方差矩阵
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;         //角速度的协方差
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;         //加速度的协方差
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;    //角速度bias的协方差
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;    //加速度bias的协方差
    
        kf_state.predict(dt, Q, in);    // IMU前向传播，每次传播的时间间隔为dt
        imu_state = kf_state.get_x();   //更新IMU状态为积分后的状态
        //更新imu上一帧的角速度 = 后一帧角速度-bias  

        last_gyr = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
        //更新imu上一帧的加速度 = R*(加速度-bias) - g
        last_acc = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();  
        last_acc = imu_state.rot * (last_acc - imu_state.ba) + imu_state.grav;
 
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    //后一个IMU时刻距离此次雷达开始的时间间隔
        imu_pose_deque.push_back( set_pose6d( offs_t, last_acc, last_gyr, imu_state.vel, imu_state.pos, imu_state.rot.matrix() ) );

    }
    // 把最后一帧IMU测量也补上
    dt = abs(pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);
    imu_state = kf_state.get_x();   

    last_imu = meas.imu.back();              //保存最后一个IMU测量，以便于下一帧使用
    last_lidar_end_time = pcl_end_time;       //保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

    /***消除每个激光雷达点的失真（反向传播）***/
    if (pcl_out.points.begin() == pcl_out.points.end()) return;
    auto it_pcl = pcl_out.points.end() - 1;  //指针变量

    M3D temp_rot;
    V3D temp_acc;
    V3D temp_vel;
    V3D temp_pos;

    //遍历imu_pose_deque队列，从最后一个IMU位姿开始，逐个修正点云的位置
    for (auto it_kp = imu_pose_deque.end() - 1; it_kp != imu_pose_deque.begin(); it_kp--)
    {
        auto head = it_kp - 1; //前一帧IMU位姿
        auto tail = it_kp;     //后一帧IMU位姿
        temp_rot<<MAT_FROM_ARRAY(head->rot);     //拿到前一帧的IMU旋转矩阵
        temp_vel<<VEC_FROM_ARRAY(head->vel);     //拿到前一帧的IMU速度
        temp_pos<<VEC_FROM_ARRAY(head->pos);     //拿到前一帧的IMU位置

        temp_acc<<VEC_FROM_ARRAY(tail->acc);     //拿到后一帧的IMU加速度
        avr_gyr<<VEC_FROM_ARRAY(tail->gyr);      //拿到后一帧的IMU角速度

        //之前点云按照时间从小到大排序过，imu_pose也同样是按照时间从小到大push进入的
        //此时从IMUpose的末尾开始循环，也就是从时间最大处开始，因此只需要判断点云时间需>IMU head时刻即可，不需要判断点云时间<IMU tail
        for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
        {
            dt = it_pcl->curvature / double(1000) - head->offset_time;    //点到IMU开始时刻的时间间隔 

            /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

            M3D R_i(temp_rot * Sophus::SO3::exp(avr_gyr * dt).matrix() );   //点it_pcl所在时刻的旋转：前一帧的IMU旋转矩阵 * exp(后一帧角速度*dt)   
        
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   //点所在时刻的位置(雷达坐标系下)
            V3D T_ei(temp_pos + temp_vel * dt + 0.5 * temp_acc * dt * dt - imu_state.pos);   //从点所在的世界位置-雷达末尾世界位置
            V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin()) break;
        }
    }

}

void ImuProcessor::process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_out)
{
    if(meas.imu.empty()) 
        return;
    ROS_ASSERT(meas.lidar != nullptr);

    if(is_need_init)
    {
        int N = 1;
        initImu(meas, kf_state, N);

        if(N > MAX_INI_COUNT)
        {
            is_need_init = false;

            //使用外部设置的参数作为初始化的协方差
            //cov_acc = cov_acc_scale;
            //cov_gyr = cov_gyr_scale;
            ROS_INFO("IMU Initial Done");

        }
        return;
    }

    undistortPointCloud(meas, kf_state, *pcl_out);

}   
