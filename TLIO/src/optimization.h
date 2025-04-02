#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Dense>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// gstam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "pcl_process.h"
#include "esekfom.hpp"
using namespace std;

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


std::mutex mtx;
std::mutex mtxLoopInfo;

/**
 * 点到坐标系原点距离
 */
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/**
 * 两点之间距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

//位姿格式变换
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}
//位姿格式变换
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}
//Eigen格式的位姿变换
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}
//Eigen格式的位姿变换
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}
//位姿格式变换 */
PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}
// 欧拉角转旋转矩阵
Eigen::Matrix3d eulerToRotationMatrix(double roll, double pitch, double yaw) {
    Eigen::Matrix3d rotation;

    // 使用 Eigen 的 AngleAxis
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    // 旋转矩阵 = Yaw * Pitch * Roll（Z-Y-X 顺序）
    rotation = yawAngle * pitchAngle * rollAngle;

    return rotation;
}
//对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn, state_ikfom& state_point)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
   // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
  // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I.matrix());       //  获取  body2lidar  外参
    T_b_lidar.pretranslate(state_point.offset_T_L_I);        

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;        

    int numberOfCores = 4;

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}




class LoopClosure{
public:

    LoopClosure(state_ikfom &state, pcl::PointCloud<PointType>::Ptr& poses3D, pcl::PointCloud<PointTypePose>::Ptr& poses6D,
    std::vector<pcl::PointCloud<PointType>::Ptr>& frames);
    ~LoopClosure();

    void setTimeStamp(double &time);
    bool detectLoopClosureDistance(int &latestID, int &closestID);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum);
    void performLoopClosure();

    bool loopClosureEnableFlag;                 //回环检测模块使能标志
    std::map<int, int> loopIndexContainer;      //闭环索引容器
    std::vector<pair<int, int>> loopIndexQueue; //闭环队列
    std::vector<gtsam::Pose3> loopPoseQueue;    //闭环位姿变换队列
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;//闭环噪声队列

    //当前状态
    state_ikfom &state_point;                   
    //历史关键帧位置3D（xyz）
    pcl::PointCloud<PointType>::Ptr &cloudKeyPoses3D;
    //历史关键帧位姿6D（欧拉角+xyz）         
    pcl::PointCloud<PointTypePose>::Ptr &cloudKeyPoses6D;
    //历史关键帧的所有平面点集合(没有降采样)
    std::vector<pcl::PointCloud<PointType>::Ptr> &surfCloudKeyFrames;    


private:
    double lidar_end_time;
    //loop clousre
    float loopClosureFrequency;                 //回环检测频率
    float historyKeyframeSearchRadius;          //回环检测匹配帧搜索半径
    float historyKeyframeSearchTimeDiff;        //回环检测匹配帧帧间时间阈值
    int historyKeyframeSearchNum;               //回环时多少个keyframe拼成submap
    float historyKeyframeFitnessScore;          //icp 匹配阈值

 
    
    pcl::VoxelGrid<PointType> downSizeFilterICP;//ICP降采样
    //kd树，用于最近邻搜索
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
};




LoopClosure::LoopClosure(state_ikfom &state, pcl::PointCloud<PointType>::Ptr& poses3D, pcl::PointCloud<PointTypePose>::Ptr& poses6D, std::vector<pcl::PointCloud<PointType>::Ptr>& frames)
:state_point(state), cloudKeyPoses3D(poses3D), cloudKeyPoses6D(poses6D), surfCloudKeyFrames(frames)
{

    lidar_end_time = 0;
    loopClosureEnableFlag = true;
    loopClosureFrequency = 4.0;
    historyKeyframeSearchRadius = 1.5;
    historyKeyframeSearchTimeDiff = 30.0;
    historyKeyframeSearchNum = 20;
    historyKeyframeFitnessScore = 0.3;

    //初始化其他成员变量
    downSizeFilterICP.setLeafSize(0.4, 0.4, 0.4);
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());


    loopIndexContainer.clear();
    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();

}

LoopClosure::~LoopClosure() {
    // 析构函数实现
}
void LoopClosure::setTimeStamp(double &time)
{
    lidar_end_time = time;
}
//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool LoopClosure::detectLoopClosureDistance(int &latestID, int &closestID)
{
    // 当前关键帧帧
    int loopKeyCur = cloudKeyPoses3D->size() - 1; //  当前关键帧索引,最后一个
    int loopKeyPre = -1;
    //当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //候选关键帧距离
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);      //历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    latestID = loopKeyCur;
    closestID = loopKeyPre;

    ROS_INFO("Find loop clousre frame ");
    return true;
}
//提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
void LoopClosure::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &cloudKeyPoses6D->points[keyNear], state_point); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

void LoopClosure::performLoopClosure()
{

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(loopKeyCur, loopKeyPre) == false)
    {
        return;
    }

    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //history keyframe submap
    {
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  将cur keyframe 转换到world系下
        // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  选取historyKeyframeSearchNum个keyframe拼成submap
        // 如果特征点较少，返回
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    //scan-to-map，调用icp匹配        
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
    //未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
        std::cout << "icp  fail!" << std::endl;
        return;
    }

    std::cout << "icp  success!" << std::endl;

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    //  闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); 
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore() ; //  loop_clousre  noise from icp
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // 添加闭环因子需要的数据
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
}


class GTSAMOptimizer{
public:

    GTSAMOptimizer(state_ikfom &state ,pcl::PointCloud<PointType>::Ptr& poses3D, pcl::PointCloud<PointTypePose>::Ptr& poses6D, std::vector<pcl::PointCloud<PointType>::Ptr>& frames);
    ~GTSAMOptimizer();

    void setInitialPose(Eigen::Vector3d &eulerAngle, Eigen::Vector3d &pos, double& last_time);
    void updatePath(const PointTypePose &pose_in, nav_msgs::Path &globalPath);
    bool iskeyFrame();
    void addOdomFactor();
    void addLoopFactor(std::vector<pair<int, int>> &loopIndexQueue, 
        std::vector<gtsam::Pose3> &loopPoseQueue, std::vector<gtsam::noiseModel::Diagonal::shared_ptr> &loopNoiseQueue);
    void recontructIKdTree(KD_TREE &ikdtree);
    void optimize(esekfom::esekf &kf, KD_TREE &ikdtree, PointCloudXYZI::Ptr &pcl_undistort, nav_msgs::Path &globalPath, std::shared_ptr<LoopClosure> &loopClosure);

    state_ikfom &state_point;//当前状态
    //历史关键帧位置3D（xyz）
    pcl::PointCloud<PointType>::Ptr &cloudKeyPoses3D;
    //历史关键帧位姿6D（欧拉角+xyz）         
    pcl::PointCloud<PointTypePose>::Ptr &cloudKeyPoses6D;
    //历史关键帧的所有平面点集合(没有降采样)
    std::vector<pcl::PointCloud<PointType>::Ptr> &surfCloudKeyFrames;    
private:

    bool isRecontructKdTree;
    int updateKdtreeCount;

    bool aLoopIsClosed;

    double lidar_end_time;
    float transformTobeMapped[6]; //  当前帧的位姿(world系下)，欧拉角和位置。

    //gtsam
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; //位姿协方差

    //Surrounding map(当前关键帧为中心，提取周围一定范围的关键帧构成的局部地图)
    float surroundingkeyframeAddingDistThreshold;  //判断是否为关键帧的距离阈值
    float surroundingkeyframeAddingAngleThreshold; //判断是否为关键帧的角度阈值
    float globalMapVisualizationSearchRadius;      //关键帧搜索半径

    pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;     //子图关键帧降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; //子图关键帧的所有点云降采样 
};

GTSAMOptimizer::GTSAMOptimizer(state_ikfom &state, pcl::PointCloud<PointType>::Ptr& poses3D, pcl::PointCloud<PointTypePose>::Ptr& poses6D, std::vector<pcl::PointCloud<PointType>::Ptr>& frames)
:state_point(state), cloudKeyPoses3D(poses3D), cloudKeyPoses6D(poses6D), surfCloudKeyFrames(frames) 
{
    isRecontructKdTree = true;
    updateKdtreeCount = 0;
    aLoopIsClosed = false;
    lidar_end_time = 0;


    surroundingkeyframeAddingDistThreshold = 1.0;
    surroundingkeyframeAddingAngleThreshold = 0.2;
    globalMapVisualizationSearchRadius = 1000.0;
    //初始化其他成员变量
    transformTobeMapped[0] = 0;
    transformTobeMapped[1] = 0;
    transformTobeMapped[2] = 0;
    transformTobeMapped[3] = 0;
    transformTobeMapped[4] = 0;
    transformTobeMapped[5] = 0;

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);
    gtSAMgraph.resize(0);
    initialEstimate.clear();
    optimizedEstimate.clear();
    isamCurrentEstimate.clear();

    downSizeFilterSubMapKeyPoses.setLeafSize(10, 10, 10);
    downSizeFilterGlobalMapKeyFrames.setLeafSize(1.0, 1.0, 1.0); 

}
GTSAMOptimizer::~GTSAMOptimizer()
{
    delete isam;
    isam = nullptr;
}
void GTSAMOptimizer::setInitialPose(Eigen::Vector3d &eulerAngle, Eigen::Vector3d &pos, double& last_time)
{
    transformTobeMapped[0] = eulerAngle(2);
    transformTobeMapped[1] = eulerAngle(1);
    transformTobeMapped[2] = eulerAngle(0);
    transformTobeMapped[3] = pos(0);
    transformTobeMapped[4] = pos(1);
    transformTobeMapped[5] = pos(2);        //清空里程计轨迹

    lidar_end_time = last_time;
}
bool GTSAMOptimizer::iskeyFrame()
{
    //如果是第一帧，直接返回,
    if(cloudKeyPoses3D->points.empty())
        return true;
    //前一帧位姿
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    //当前帧位姿
    Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);

    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    //获取上一帧相对当前帧的位姿
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); 
    
    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;

    return true;
}
//添加里程计因子
void GTSAMOptimizer::addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        //第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        //变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        //添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); //前一帧
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   //当前帧
        //参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        //变量节点设置初始值
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}
//添加闭环因子
void GTSAMOptimizer::addLoopFactor(std::vector<pair<int, int>> &loopIndexQueue,
                                   std::vector<gtsam::Pose3> &loopPoseQueue,
                                   std::vector<gtsam::noiseModel::Diagonal::shared_ptr> &loopNoiseQueue)
{
    if (loopIndexQueue.empty())
        return;

    //闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        //闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //当前帧
        int indexTo = loopIndexQueue[i].second;  //之前的帧
        //闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}


void GTSAMOptimizer::recontructIKdTree(KD_TREE &ikdtree){
    if(isRecontructKdTree  &&  updateKdtreeCount >  0){
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap的pose集合

        downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
        downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);         //  subMap poses  downsample
        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd], state_point); //  fast_lio only use  surfCloud
        }
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

        ikdtree.reconstruct(subMapKeyFramesDS->points);
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed  ikdtree ");
    }
    updateKdtreeCount ++ ; 
}

void GTSAMOptimizer::updatePath(const PointTypePose &pose_in, nav_msgs::Path &globalPath)
{
    string odometryFrame = "camera_init";
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}
//保存关键帧和因子
void GTSAMOptimizer::optimize(esekfom::esekf &kf, KD_TREE &ikdtree, PointCloudXYZI::Ptr &pcl_undistort, nav_msgs::Path &globalPath, std::shared_ptr<LoopClosure> &loopClosure)
{
    //计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (iskeyFrame() == false)
        return;
    //激光里程计因子(from fast-lio),  输入的是前后帧间位姿(Lidar系下)
    addOdomFactor();
    //闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor(loopClosure->loopIndexQueue, loopClosure->loopPoseQueue, loopClosure->loopNoiseQueue);
    //执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (aLoopIsClosed == true) // 有回环因子，多update几次
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    //update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    //优化结果
    isamCurrentEstimate = isam->calculateBestEstimate();
    //当前帧位姿结果
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); //使用intensity作为该帧点云的index
    cloudKeyPoses3D->push_back(thisPose3D);         //将gtsam优化的关键帧位置放入队列中

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);         //将gtsam优化的关键帧位姿放入队列中   

    //位姿协方差
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    //ESKF状态和方差更新
    ////////////////////////////////////////需要修改，s-fast-lio用的是sophus
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Matrix3d rot = eulerToRotationMatrix(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());
    //更新状态
    state_ikfom state_updated = kf.get_x(); //获取当前状态
    state_updated.pos = pos;
    state_updated.rot = Sophus::SO3(rot);
    kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正

    // TODO:P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); //获取当前的状态估计的协方差矩阵
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);

    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*pcl_undistort, *thisSurfKeyFrame); 
    surfCloudKeyFrames.push_back(thisSurfKeyFrame); //存储关键帧,没有降采样的所有点云


    updatePath(thisPose6D, globalPath); //  可视化update后的path

    //如果回环
    if (aLoopIsClosed == true)
    {
        //清空里程计轨迹
        globalPath.poses.clear();
        //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();
            updatePath(cloudKeyPoses6D->points[i], globalPath);
        }
        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree(ikdtree);
        ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}




#endif