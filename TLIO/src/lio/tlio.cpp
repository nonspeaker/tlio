#include <thread>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Core>

#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

// 自定义头文件
#include "imu_process.h"
#include "pcl_process.h"
#include "local_map.h"
#include "common_lib.hpp"
#include "message_receiver.h"
#include "message_publisher.h"



using namespace std;


/******************ROS 配置参数 ******************** */
//common
std::string lidar_topic = "/livox/lidar";
std::string imu_topic = "/livox/imu";
bool time_sync_en = false;
double time_offset_lidar_to_imu = 0.0;//lidar相对于imu时间偏移，配置参数
//preprocess
int lidar_type = AVIA;
int scan_line = 16;
int scan_rate = 10;
int time_unit = SEC;
double blind = 0.01;

//mapping
double acc_cov = 0.1;
double gyr_cov = 0.1;
double b_acc_cov = 0.0001;
double b_gyr_cov = 0.0001;
double fov_deg = 360;
float det_range = 300.0f;
bool extrinsic_est_en = true;
std::vector<double> extrinT = vector<double>{-0.011, -0.02329, 0.04412};
std::vector<double> extrinR = vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
//publish
bool path_en = true;
bool scan_pub_en = true;
bool dense_pub_en = true;
bool scan_body_pub_en = true;
//pcd_save
bool pcd_save_en = false;
int pcd_save_interval = -1;
// voxel filter paprams
float odometrySurfLeafSize = 0.4;
float mappingCornerLeafSize = 0.2;
float mappingSurfLeafSize = 0.4;


// launch file
bool feature_enabled = false;
int point_filter_num = 2;
int max_iteration = 4;
double filter_size_surf_min = 0.5;
double filter_size_map_min = 0.5;
double cube_len = 200;
bool runtime_pos_log_enable = false;

double filter_publish_map = 0.5;
/***************************************************** */


condition_variable sig_buffer;


double lidar_end_time = 0.0;

bool is_first_scan = true;
bool is_ekf_init = false;
double first_lidar_time = 0.0;
int feats_down_size = 0;

MeasureGroup Measures;//当前雷达测量数据包
esekfom::esekf kf;//卡尔曼滤波器状态
state_ikfom state_point;//当前状态
Eigen::Matrix<double, 24, 24>  P; //协方差矩阵
Eigen::Vector3d lidar_position;//当前雷达位置

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());//当前帧去畸变后的点云
PointCloudXYZI::Ptr feats_down_lidar(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云pcd_index

PointCloudXYZI::Ptr feats_world(new PointCloudXYZI()); // 世界坐标系的点云
PointCloudXYZI::Ptr feats_imu(new PointCloudXYZI());   // IMU坐标系的点云

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());//保存的点云

std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());
std::shared_ptr<ImuProcessor> imuProcessor(new ImuProcessor());
std::shared_ptr<LocalMapManager> localMapManager(new LocalMapManager());

KD_TREE ikdtree;
std::vector<PointVector> Nearest_Points;


nav_msgs::Path globalPath; //优化后的全局路径
visualization_msgs::MarkerArray markerArray;//回环检测约束

bool flg_exit = false;
//信号处理函数，用于捕获退出信号  @param sig 信号编号
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}


void loadParameters(ros::NodeHandle& nh) {

    nh.param<string>("common/lidar_topic", lidar_topic, "/livox/lidar");                //雷达点云话题
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");                      //IMU话题
    nh.param<bool>("common/time_sync_en", time_sync_en, false);                         //是否开启时间同步
    nh.param<double>("common/time_offset_lidar_to_imu", time_offset_lidar_to_imu, 0.0); //雷达相对于IMU时间偏移
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);                           //雷达类型  
    std::cout << "p_pre->lidar_type " << lidar_type << std::endl;           
    nh.param<int>("preprocess/scan_line", scan_line, 16);                               //激光雷达线数
    nh.param<int>("preprocess/scan_rate", scan_rate, 10);
    nh.param<double>("preprocess/blind", blind, 0.01);                                  //盲区
    nh.param<int>("preprocess/timestamp_unit", time_unit, US);                          //时间单位

    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);                                  //加速度计噪声协方差
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);                                  //陀螺仪噪声协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);                           //加速度计偏置噪声协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);                           //陀螺仪偏置噪声协方差
    nh.param<double>("mapping/fov_degree", fov_deg, 180);                               //视场角度
    nh.param<float>("mapping/det_range", det_range, 300.f);                             //检测范围  
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);                 //是否估计外参
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());         //激光雷达相对于IMU的平移
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());         //激光雷达相对于IMU的旋转

    nh.param<bool>("publish/path_en", path_en, true);                                   //是否发布路径
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);                       //是否发布点云
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);                     //是否发布稠密点云
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);            //是否发布点云在雷达坐标系下的点云

    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);                         //是否保存点云  
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);                          //点云保存间隔


    nh.param<bool>("feature_extract_enable", feature_enabled, false);                   //特征提取开关
    nh.param<int>("point_filter_num", point_filter_num, 2);                             //点云滤波器数量
    nh.param<int>("max_iteration", max_iteration, 4);                                   //最大迭代次数
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);                    //平面点滤波器大小
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);                      //地图滤波大小
    nh.param<double>("cube_side_length", cube_len, 200);                                //地图立方体边长
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_enable, 0);                //是否启用运行时位置日志

    nh.param<double>("filter_publish_map", filter_publish_map, 0.5);                    //发布点云大小

}
void initializeProcessors() {

    pclProcessor->setParams(lidar_type, scan_line, scan_rate, time_unit, blind, feature_enabled, point_filter_num);
    localMapManager->setParams(det_range, filter_size_map_min, cube_len);

    //IMU处理器参数
    Eigen::Vector3d Lidar_T_wrt_IMU = Eigen::Vector3d::Zero();     //激光雷达相对于IMU的平移
    Eigen::Matrix3d Lidar_R_wrt_IMU = Eigen::Matrix3d::Identity(); //激光雷达相对于IMU的旋转
    
    // 从配置参数中加载外参
    Lidar_T_wrt_IMU << extrinT[0], extrinT[1], extrinT[2];
    Lidar_R_wrt_IMU << extrinR[0], extrinR[1], extrinR[2],
                       extrinR[3], extrinR[4], extrinR[5],
                       extrinR[6], extrinR[7], extrinR[8];
    imuProcessor->set_params(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, Eigen::Vector3d(gyr_cov, gyr_cov, gyr_cov), Eigen::Vector3d(acc_cov, acc_cov, acc_cov),
        Eigen::Vector3d(b_gyr_cov, b_gyr_cov, b_gyr_cov), Eigen::Vector3d(b_acc_cov, b_acc_cov, b_acc_cov));

}


int main(int argc, char** argv) 
{
    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;

    loadParameters(nh);

    initializeProcessors();

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);//一秒执行5000次

    MessagePublisher* publisher = new MessagePublisher(nh);
    MessageReceiver* receiver = new MessageReceiver(nh, lidar_topic, imu_topic, pclProcessor);

    receiver->setParameters(time_sync_en, time_offset_lidar_to_imu);
    while(ros::ok())
    {

        if (flg_exit) 
            break;
        ros::spinOnce();//处理一次回调函数

        if(receiver->syncPackages(Measures, lidar_end_time))
        {
            if(is_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                is_first_scan = false;
                continue;
            }       

            //std::cout << "feats_raw_size: " << Measures.lidar->points.size()  << std::endl;
            
            //点云去运动畸变，反向传播
            imuProcessor->process(Measures, kf, feats_undistort);
            int feats_undistort_size = feats_undistort->points.size();
            //std::cout << "feats_undistort_size: " << feats_undistort_size << std::endl;
            

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //获取雷达当前位置，更新局部地图范围，删除k-d树中超出范围的点。
            state_point = kf.get_x();
            lidar_position = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;
            //检查当前lidar数据时间，与最早lidar数据时间是否足够//判断EKF是否初始化，根据当前雷达数据包的时间与第一帧雷达数据包的时间戳的差值是否小于初始化时间
            is_ekf_init = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            //根据lidar在世界坐标系下的位置，重新确定局部地图范围，移除距离远的点。
            localMapManager->updateLocalMapRange(lidar_position, ikdtree);



            //下采样得到当前帧的点云
            pclProcessor->downsamplePointCloud(feats_undistort, feats_down_lidar, filter_size_surf_min);
            feats_down_size = feats_down_lidar->points.size();


            //当前帧点云数量少，则警告
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //初始化k-d树，存第一帧点云
            if (ikdtree.Root_Node == nullptr) {
                localMapManager->initializeKdTree(ikdtree, feats_down_lidar, state_point);
                continue;
            }

            //迭代状态估计
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_lidar, ikdtree, Nearest_Points, max_iteration, extrinsic_est_en);

            //发布里程计信息
            state_point = kf.get_x();
            P = kf.get_P();
            publisher->publishOdometry(state_point, P, lidar_end_time);
            //发布路径
            publisher->publishPath(state_point, lidar_end_time);


            //向地图k-d树里添加点云
            localMapManager->updateMapIncremental(feats_down_lidar, ikdtree, Nearest_Points, state_point, is_ekf_init);

            //发布点云
            //发布IMU坐标系下的点云
            localMapManager->transformToIMU(feats_undistort, feats_imu, state_point);
            publisher->publishPointCloudIMU(feats_imu, lidar_end_time);
            //发布世界坐标系下的点云
            localMapManager->transformToWorld(feats_undistort, feats_world, state_point);
            pclProcessor->downsamplePointCloud(feats_world, feats_world, filter_publish_map);
            publisher->publishPointCloudWorld(feats_world, lidar_end_time);

            //增量保存全部点云
            *pcl_wait_save += *feats_world;
        }
        rate.sleep();
    }

    //保存里程计
    publisher->writeOdometryToFile();

    //保存点云
    if(pcd_save_en)
    {
        std::string savePath(string(string(ROOT_DIR) + "PCD/scans") + string(".pcd"));
        localMapManager->savePointCloud(pcl_wait_save, savePath);
    }


    return 0;
}