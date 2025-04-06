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
#include "optimization.h"
#include "common_lib.hpp"
#include "message_receiver.h"
#include "message_publisher.h"



using namespace std;


/******************ROS 配置参数 ******************** */
//common
std::string lid_topic = "/livox/lidar";
std::string imu_topic = "/livox/imu";
bool time_sync_en = false;
double time_offset_lidar_to_imu = 0.0;//lidar相对于imu时间偏移，配置参数
//preprocess
int lidar_type = AVIA;
int scan_line = 16;
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
// robot motion constraint (in case you are using a 2D robot)
float z_tollerance = 1000;
float rotation_tollerance = 1000;
// CPU Params
int numberOfCores = 4;
double mappingProcessInterval = 0.15;
// Surrounding map
float surroundingkeyframeAddingDistThreshold = 1.0;
float surroundingkeyframeAddingAngleThreshold = 0.2;  
float surroundingKeyframeDensity = 2.0;              
float surroundingKeyframeSearchRadius = 50.0;      
// Loop closure
bool loopClosureEnableFlag = true;                  
float loopClosureFrequency = 4.0;                    
int surroundingKeyframeSize = 50;                 
float historyKeyframeSearchRadius = 1.5;            
float historyKeyframeSearchTimeDiff = 30.0;           
int historyKeyframeSearchNum = 20;               
float historyKeyframeFitnessScore = 0.3;
// Visualization
float globalMapVisualizationSearchRadius = 1000.0;    
float globalMapVisualizationPoseDensity = 10;     
float globalMapVisualizationLeafSize = 1.0;       
// visual iktree_map  
bool visulize_IkdtreeMap = true;
bool recontructKdTree = true;
// Export settings
bool savePCD = true;
std::string savePCDDirectory;


// launch file
bool feature_extracted_enable = false;
int point_filter_num = 2;
int max_iteration = 4;
double filter_size_surf_min = 0.5;
double filter_size_map_min = 0.5;
double cube_len = 200;
bool runtime_pos_log_enable = false;

/***************************************************** */


condition_variable sig_buffer;

//历史关键帧位置3D（xyz）
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());   
//历史关键帧位姿6D（欧拉角+xyz）         
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); 
//历史关键帧的所有平面点集合(没有降采样)
std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;    


double lidar_end_time = 0.0;

bool is_first_scan = true;
bool is_ekf_init = false;
double first_lidar_time = 0.0;
int feats_down_size = 0;

MeasureGroup Measures;//当前雷达测量数据包
esekfom::esekf kf;//卡尔曼滤波器状态
state_ikfom state_point;//当前状态
Eigen::Vector3d lidar_position;//当前雷达位置

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());//当前帧去畸变后的点云
PointCloudXYZI::Ptr feats_down_lidar(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云pcd_index
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云（世界坐标系）
PointCloudXYZI::Ptr feats_publish(new PointCloudXYZI());//发布出去的点云


std::shared_ptr<GTSAMOptimizer> gtsamOptimizer(new GTSAMOptimizer(state_point, cloudKeyPoses3D, cloudKeyPoses6D, surfCloudKeyFrames));
std::shared_ptr<LoopClosure> loopClosure(new LoopClosure(state_point, cloudKeyPoses3D, cloudKeyPoses6D, surfCloudKeyFrames));
std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());
std::shared_ptr<ImuProcessor> imuProcessor(new ImuProcessor());

KD_TREE ikdtree;
std::vector<PointVector> Nearest_Points;

nav_msgs::Path path;//lidar移动路径信息的消息
nav_msgs::Path globalPath; //优化后的全局路径
nav_msgs::Odometry odomAftMapped;//建图后里程计的消息
visualization_msgs::MarkerArray markerArray;

bool flg_exit = false;
//信号处理函数，用于捕获退出信号  @param sig 信号编号
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}


//回环检测线程
void loopClosureThread(MessagePublisher &publisher)
{
    if (loopClosure->loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok())
    {
        rate.sleep();
        loopClosure->setTimeStamp(lidar_end_time);
        loopClosure->performLoopClosure();   //回环检测
        loopClosure->generateLoopMarkers(markerArray, lidar_end_time);

        publisher.publishLoopConstraints(markerArray);
    }
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;

    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");                    //雷达点云话题
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");                      //IMU话题
    nh.param<bool>("common/time_sync_en", time_sync_en, false);                         //是否开启时间同步
    nh.param<double>("common/time_offset_lidar_to_imu", time_offset_lidar_to_imu, 0.0); //雷达相对于IMU时间偏移
    nh.param<int>("preprocess/lidar_type", pclProcessor->lidar_type, AVIA);             //雷达类型  
    std::cout << "p_pre->lidar_type " << pclProcessor->lidar_type << std::endl;           
    nh.param<int>("preprocess/scan_line", pclProcessor->scan_line, 16);                 //激光雷达线数
    nh.param<double>("preprocess/blind", pclProcessor->blind, 0.01);                    //盲区
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
    //voxel filter paprams
    nh.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);                 //odometry滤波器大小 
    nh.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);               //mapping角点滤波器大小
    nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);                   //mapping平面点滤波器大小
    //robot motion constraint (in case you are using a 2D robot)
    nh.param<float>("z_tollerance", z_tollerance, FLT_MAX);                             //z轴容差
    nh.param<float>("rotation_tollerance", rotation_tollerance, FLT_MAX);               //旋转容差
    //CPU Params
    nh.param<int>("numberOfCores", numberOfCores, 2);                                   //CPU核心数
    nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);           //建图处理间隔
    //Surrounding map
    nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);    //添加关键帧的距离阈值
    nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);   //添加关键帧的角度阈值
    nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);                             //关键帧的稀疏程度
    nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);                  //搜索周围关键帧的半径
    // loop clousre
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);                   //回环检测使能
    nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);                      //回环检测频率
    nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);                   //搜索关键帧的数量
    nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);       //历史关键帧搜索半径
    nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);   //历史关键帧搜索时间差
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);                 //历史关键帧搜索数量
    nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);        //历史关键帧拟合分数
    // Visualization
    nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3); //全局地图可视化搜索半径
    nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);  //全局地图可视化点密度
    nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);         //全局地图可视化体素滤波器大小
    // visual ikdtree map
    nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);               //是否可视化ikdtree地图
    // reconstruct ikdtree  
    nh.param<bool>("recontructKdTree", recontructKdTree, false);                     //是否重建ikdtree
    // savMap
    nh.param<bool>("savePCD", savePCD, false);                                       //是否保存点云
    nh.param<std::string>("savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/"); //保存点云目录0


    nh.param<bool>("feature_extract_enable", pclProcessor->feature_enabled, false);  //特征提取开关
    nh.param<int>("point_filter_num", pclProcessor->point_filter_num, 2);            //点云滤波器数量
    nh.param<int>("max_iteration", max_iteration, 4);                                //最大迭代次数
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);                 //平面点滤波器大小
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);                   //地图滤波器大小
    nh.param<double>("cube_side_length", cube_len, 200);                             //地图立方体边长
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_enable, 0);             //是否启用运行时位置日志

    //初始化地图时间戳和帧
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

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

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);//一秒执行5000次

    MessagePublisher publisher(nh);

    MessageReceiver receiver(nh, pclProcessor);
 
    // 回环检测线程
    std::thread loopthread(loopClosureThread, std::ref(publisher));
    while(ros::ok())
    {

        if (flg_exit) 
            break;
        ros::spinOnce();//处理一次回调函数

        if(receiver.syncPackages(Measures, lidar_end_time))
        {
            if(is_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                is_first_scan = false;
                continue;
            }       

            std::cout << "feats_raw_size: " << Measures.lidar->points.size()  << std::endl;
            
            //点云去运动畸变，反向传播
            imuProcessor->process(Measures, kf, feats_undistort);
            int feats_undistort_size = feats_undistort->points.size();
            std::cout << "feats_undistort_size: " << feats_undistort_size << std::endl;
            

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //获取雷达当前位置，更新局部地图范围，删除k-d树中超出范围的点。
            state_point = kf.get_x();
            lidar_position = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            //检查当前lidar数据时间，与最早lidar数据时间是否足够//判断EKF是否初始化，根据当前雷达数据包的时间与第一帧雷达数据包的时间戳的差值是否小于初始化时间
            is_ekf_init = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            //根据lidar在世界坐标系下的位置，重新确定局部地图范围，移除距离远的点。
            pclProcessor->updateLocalMapRange(lidar_position, ikdtree);
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
                pclProcessor->initializeKdTree(ikdtree, feats_down_lidar, feats_down_world, filter_size_map_min, state_point);
                continue;
            }

            //迭代状态估计
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_lidar, ikdtree, Nearest_Points, max_iteration, extrinsic_est_en);
            state_point = kf.get_x();


           //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
            Eigen::Vector3d eulerAngle = state_point.rot.matrix().eulerAngles(2,1,0); 
            gtsamOptimizer->setInitialPose(eulerAngle,state_point.pos, lidar_end_time);
            gtsamOptimizer->optimize(kf, ikdtree, feats_undistort, globalPath, loopClosure);
 

            //向地图k-d树里添加点云
            feats_down_world->resize(feats_down_size);
            pclProcessor->updateMapIncremental(feats_down_lidar, feats_down_world, ikdtree, Nearest_Points, state_point);

            publisher.publishPath(path, state_point, lidar_end_time);

            pclProcessor->transformToWorld(feats_undistort, feats_publish, state_point);
            pclProcessor->downsamplePointCloud(feats_publish, feats_publish, filter_size_map_min);
            publisher.publishPointCloud(feats_publish, lidar_end_time);

            std::cout << "feats_down_size: " << feats_down_size << std::endl;

        }
        rate.sleep();

    }

    return 0;
}