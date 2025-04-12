#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

// 自定义头文件
#include <ikd-Tree/ikd_Tree.h>
#include "use-ikfom.hpp"
#include "esekfom.hpp"
#include "common_lib.hpp"

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
    void setParams(bool enableFlag, float loopClosureFreq, float searchRadius, float searchTimeDiff, int searchNum, float fitnessScore);
    bool detectLoopClosureDistance(int &latestID, int &closestID);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum);
    void generateLoopMarkers(visualization_msgs::MarkerArray &markerArray, double &lidar_end_time);
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




class GTSAMOptimizer{
public:

    GTSAMOptimizer(state_ikfom &state ,pcl::PointCloud<PointType>::Ptr& poses3D, pcl::PointCloud<PointTypePose>::Ptr& poses6D, std::vector<pcl::PointCloud<PointType>::Ptr>& frames);
    ~GTSAMOptimizer();
    void setParams(bool reconstructKdTree, float distThreshold, float angleThreshold, float searchRadius, float poseDensity, float leafSize);
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

    int updateKdtreeCount;

    bool aLoopIsClosed;

    double lidar_end_time;
    float transformTobeMapped[6]; //  当前帧的位姿(world系下)，欧拉角和位置。

    //gtsam
    gtsam::ISAM2 *isam;
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; //位姿协方差

    bool isRecontructKdTree;
    //Surrounding map(当前关键帧为中心，提取周围一定范围的关键帧构成的局部地图)
    float surroundingkeyframeAddingDistThreshold;  //判断是否为关键帧的距离阈值
    float surroundingkeyframeAddingAngleThreshold; //判断是否为关键帧的角度阈值

    float globalMapVisualizationSearchRadius;      //关键帧搜索半径

    pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;     //子图关键帧降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; //子图关键帧的所有点云降采样 
};




#endif