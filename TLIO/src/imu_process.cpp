#include "imu_process.h"

ImuProcessor::ImuProcessor()
{
    is_need_init = true;    
    is_first_frame = true; 
    init_iter_num = 1;          

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
void ImuProcessor::Reset()   //重置参数
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  last_gyr = V3D(0, 0, 0);
  is_need_init = true;                   //是否需要初始化imu

  init_iter_num = 1;                       //初始化迭代次数
  imu_pose_deque.clear();                         // imu位姿清空
  last_imu.reset(new sensor_msgs::Imu()); //上一帧imu初始化

}

//初始化IMU：初始化卡尔曼滤波器
 void ImuProcessor::initImu(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
 {
    V3D cur_acc;
    V3D cur_gyr;

    if (is_first_frame) //如果为第一帧IMU
    {
      Reset();    //重置IMU参数
      N = 1;      //将迭代次数置1
      is_first_frame = false;
      const auto &imu_acc = meas.imu.front()->linear_acceleration;    //IMU初始时刻的加速度
      const auto &gyr_acc = meas.imu.front()->angular_velocity;       //IMU初始时刻的角速度
      mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //第一帧加速度值作为初始化均值
      mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              //第一帧角速度值作为初始化均值
    }

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
    std::cout << "grav: " << init_state.grav.transpose() << std::endl;
    init_state.bg = mean_gyr;                                   //角速度偏置（测量的平均角速度）
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;                  //Lidar相对于IMU的平移外参
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);     //Lidar相对于IMU的旋转外参
    kf_state.change_x(init_state);                              //将初始化后的状态量赋值给卡尔曼滤波器

    Eigen::Matrix<double, 24, 24> init_P = Eigen::MatrixXd::Identity(24, 24);  //初始化协方差矩阵
    init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
    init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
    init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
    init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
    init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
    kf_state.change_P(init_P);  //将初始化后的协方差矩阵赋值给卡尔曼滤波器

    last_imu = meas.imu.back(); //更新上一个IMU数据
}


void ImuProcessor::preintegrateIMU(const MeasureGroup &meas, esekfom::esekf &kf_state) {
    auto imu_deque = meas.imu;   // 获取 IMU 队列
    imu_deque.push_front(last_imu); // 将上一帧尾部的 IMU 数据插入到 IMU 队列的开头

    const double &imu_end_time = imu_deque.back()->header.stamp.toSec(); // 当前帧尾部的 IMU 时间戳
    const double &pcl_beg_time = meas.lidar_beg_time;      // 点云开始的时间戳
    const double &pcl_end_time = meas.lidar_end_time;      //点云结束的时间戳

    state_ikfom imu_state = kf_state.get_x();  // 获取上一次 KF 估计的后验状态作为本次 IMU 预测的初始状态
    imu_pose_deque.clear(); // 清空 IMU 位姿队列
    imu_pose_deque.push_back(set_pose6d(0.0, last_acc, last_gyr, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));

    V3D avr_acc; // 平均加速度
    V3D avr_gyr; // 平均角速度

    double dt = 0;
    input_ikfom in;

    // 遍历 IMU 队列，计算 IMU 位姿
    for (auto it_imu = imu_deque.begin(); it_imu < (imu_deque.end() - 1); it_imu++) {
        auto &&head = *(it_imu);        // 当前 IMU 数据
        auto &&tail = *(it_imu + 1);    // 下一帧 IMU 数据

        if (tail->header.stamp.toSec() < last_lidar_end_time) 
            continue;

        avr_gyr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), //中值积分
                   0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                   0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        avr_acc << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                   0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                   0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        avr_acc = avr_acc * G_m_s2 / mean_acc.norm(); // 调整加速度

        if (head->header.stamp.toSec() < last_lidar_end_time) {
            dt = tail->header.stamp.toSec() - last_lidar_end_time;
        } else {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        in.acc = avr_acc;
        in.gyro = avr_gyr;

        // 配置协方差矩阵
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

        kf_state.predict(dt, Q, in);    // IMU 前向传播
        imu_state = kf_state.get_x();   // 更新 IMU 状态

        last_gyr = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
        last_acc = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();
        last_acc = imu_state.rot * (last_acc - imu_state.ba) + imu_state.grav;

        double offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        imu_pose_deque.push_back(set_pose6d(offs_t, last_acc, last_gyr, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    }

    // 补上最后一帧 IMU 测量
    dt = abs(pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);

    last_imu = meas.imu.back();          // 保存最后一个 IMU 测量
    last_lidar_end_time = pcl_end_time; // 保存雷达结束时间
}


void ImuProcessor::removeDistortionByIMU(PointCloudXYZI &pcl_out, esekfom::esekf &kf_state) {
    if (pcl_out.points.empty()) return;

    auto it_pcl = pcl_out.points.end() - 1;  // 从点云末尾开始
    state_ikfom imu_state = kf_state.get_x();

    M3D rot_prev;  // 前一个旋转矩阵
    V3D vel_prev;  // 前一个速度
    V3D pos_prev;  // 前一个位置
    
    V3D acc_next;  // 后一个加速度
    V3D gyr_next;  // 后一个角速度

    // 遍历 imu_pose_deque，从最后一个 IMU 位姿开始修正点云
    for (auto it_kp = imu_pose_deque.end() - 1; it_kp != imu_pose_deque.begin(); it_kp--) {
        auto head = it_kp - 1; // 前一帧 IMU 位姿
        auto tail = it_kp;     // 后一帧 IMU 位姿

        rot_prev << MAT_FROM_ARRAY(head->rot);
        vel_prev << VEC_FROM_ARRAY(head->vel);
        pos_prev << VEC_FROM_ARRAY(head->pos);

        acc_next << VEC_FROM_ARRAY(tail->acc);
        gyr_next << VEC_FROM_ARRAY(tail->gyr);

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            double dt = it_pcl->curvature / double(1000) - head->offset_time;

            M3D R_i(rot_prev * Sophus::SO3::exp(gyr_next * dt).matrix());
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_prev + vel_prev * dt + 0.5 * acc_next * dt * dt - imu_state.pos);
            V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() *
                               (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

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
        initImu(meas, kf_state, init_iter_num);

        is_need_init = true;
        if(init_iter_num > MAX_INI_COUNT)
        {
            is_need_init = false;
            cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      
            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;

            ROS_INFO("IMU Initial Done");

        }
        return;
    }

    // 将 meas.lidar 的值赋给 pcl_out
    pcl_out = meas.lidar;

    // 对点云进行重排序
    sort(pcl_out->points.begin(), pcl_out->points.end(), time_list); //curvature 中存放了时间戳

    // IMU预积分
    preintegrateIMU(meas, kf_state);

    // 点云去畸变
    removeDistortionByIMU(*pcl_out, kf_state);

}   