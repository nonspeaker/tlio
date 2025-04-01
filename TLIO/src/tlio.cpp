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

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <livox_ros_driver2/CustomMsg.h>


#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

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

#include <condition_variable>

#include "imu_process.h"
#include "pcl_process.h"


#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

using namespace std;


enum LID_TYPE
{
  AVIA = 1,
  VELO16,
  OUST64,
  RS32
}; //{1, 2, 3, 4}
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};

bool path_en = true;
bool scan_pub_en = true;
bool dense_pub_en = true;
bool scan_body_pub_en = true;

int NUM_MAX_ITERATIONS = 4;
string map_file_path = "";

string lid_topic = "/livox/lidar";
string imu_topic = "/livox/imu";
bool time_sync_en = false;
double time_offset_lidar_to_imu = 0.0;//lidar相对于imu时间偏移，配置参数

double filter_size_corner_min = 0.5;
double filter_size_surf_min = 0.5;
double filter_size_map_min = 0.5;
double cube_len = 200;

float DET_RANGE = 300.0f;
double fov_deg = 360;
double gyr_cov = 0.1;
double acc_cov = 0.1;
double b_gyr_cov = 0.0001;
double b_acc_cov = 0.0001;
vector<double> extrinT = vector<double>{-0.011, -0.02329, 0.04412};
vector<double> extrinR = vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
bool extrinsic_est_en = true;

double blind = 0.01;
int lidar_type = AVIA;
int timestamp_unit = US;
int N_SCANS = 16;
int SCAN_RATE = 10;

int point_filter_num = 2;
bool feature_extracted_enable = false;

bool pcd_save_en = true;
int pcd_save_interval = -1;

bool timediff_set_flg = false;
double timediff_lidar_to_imu = 0.0;
double last_timestamp_lidar = 0.0;
double last_timestamp_imu = -1.0;



mutex mtx_buffer;
condition_variable sig_buffer;
std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<double> time_buffer;
std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());
bool lidar_pushed;
double lidar_end_time = 0.0;
double lidar_mean_scan_time = 0.1; //用于同步

bool is_first_scan = true;
bool is_ekf_init = false;
double first_lidar_time = 0.0;
int feats_down_size = 0;

MeasureGroup Measures;//当前雷达测量数据包
esekfom::esekf kf;//卡尔曼滤波器状态
state_ikfom state_point;//当前状态
Eigen::Vector3d lidar_position;//当前雷达位置
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());//地图中的特征点点云
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());//当前帧去畸变后的点云
PointCloudXYZI::Ptr feats_down_lidar(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云pcd_index
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云（世界坐标系）

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

KD_TREE ikdtree;
int kdtree_delete_counter = 0;
bool is_localmap_init = false;
const float MOV_THRESHOLD = 1.5f;
BoxPointType localmap_range;
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 


nav_msgs::Path path;//lidar移动路径信息的消息
nav_msgs::Odometry odomAftMapped;//建图后里程计的消息
geometry_msgs::PoseStamped msg_body_pose;//机体位姿及其时间戳的消息

/*back end*/
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   //历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         //历史关键帧位置3D（xyz）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //历史关键帧位姿6D（欧拉角+xyz）
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

pcl::PointCloud<PointTypePose>::Ptr fastlio_unoptimized_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  存储fastlio 未优化的位姿
pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  gnss 轨迹


// voxel filter paprams
float odometrySurfLeafSize = 0.4;
float mappingCornerLeafSize = 0.2;
float mappingSurfLeafSize = 0.4;

float z_tollerance = 1000;
float rotation_tollerance = 1000;

// CPU Params
int numberOfCores = 4;
double mappingProcessInterval = 0.15;


//loop clousre
bool startFlag = true;
bool loopClosureEnableFlag = true;
float loopClosureFrequency = 4.0; //   回环检测频率
int surroundingKeyframeSize = 50;
float historyKeyframeSearchRadius = 1.5;   // 回环检测 radius kdtree搜索半径
float historyKeyframeSearchTimeDiff = 30.0; //  帧间时间阈值
int historyKeyframeSearchNum = 20;        //   回环时多少个keyframe拼成submap
float historyKeyframeFitnessScore = 0.3;   // icp 匹配阈值
bool potentialLoopFlag = false;

ros::Publisher pubHistoryKeyFrames; //  发布 loop history keyframe submap
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue; //闭环队列
vector<gtsam::Pose3> loopPoseQueue;    //闭环位姿变换队列
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;

nav_msgs::Path globalPath;//全局路径

// 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

//点云降采样
pcl::VoxelGrid<PointType> downSizeFilterSurf; //lidar点云
pcl::VoxelGrid<PointType> downSizeFilterICP;  //ICP
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; //周围关键帧，用于帧到地图优化


float transformTobeMapped[6]; //  当前帧的位姿(world系下)，欧拉角和位置。
std::mutex mtx;
std::mutex mtxLoopInfo;

//Surrounding map(当前关键帧为中心，提取周围一定范围的关键帧构成的局部地图)
float surroundingkeyframeAddingDistThreshold = 1.0;  //  判断是否为关键帧的距离阈值
float surroundingkeyframeAddingAngleThreshold = 0.2; //  判断是否为关键帧的角度阈值
float surroundingKeyframeDensity = 2.0;
float surroundingKeyframeSearchRadius = 50.0;

//gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap ;           //发布最后优化的地图

bool isRecontructKdTree = false;
int updateKdtreeCount = 0 ;        //  每100次更新一次
bool visulize_IkdtreeMap = false;            //  visual iktree submap

//global map visualization radius
float globalMapVisualizationSearchRadius = 1000.0;
float globalMapVisualizationPoseDensity = 10;
float globalMapVisualizationLeafSize = 1.0;


bool flg_exit = false;

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

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
////////////////////////////////////////////////////////




//发布thisCloud，返回thisCloud对应msg格式
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}


//点到坐标系原点距离
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

//两点之间距离
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}


/////////////////////////////////////////////////////

//欧拉角转旋转矩阵
Eigen::Matrix3d eulerToRotationMatrix(float roll, float pitch, float yaw) {
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    // 按 ZYX 顺序组合旋转
    Eigen::Quaterniond quaternion = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = quaternion.toRotationMatrix();
    return rotationMatrix;
}
// 将更新的pose赋值到transformTobeMapped
void getCurPose(state_ikfom cur_state)
{
    //欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
    //V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw
    
    transformTobeMapped[0] = eulerAngle(2);                //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);                //  pitch
    transformTobeMapped[2] = eulerAngle(0);                //  yaw
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}
//计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
bool iskeyFrame()
{
    //历史关键帧位置
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // 当前帧位姿
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
void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        // 变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); //前一帧
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   //当前帧
        // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}
//添加闭环因子
void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //当前帧
        int indexTo = loopIndexQueue[i].second;  //之前的帧
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}
//更新里程计轨迹
void updatePath(const PointTypePose &pose_in)
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
void saveKeyFramesAndFactor()
{
    //计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (iskeyFrame() == false)
        return;
    //激光里程计因子(from fast-lio),  输入的是前后帧间位姿(Lidar系下)
    addOdomFactor();
    //闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();
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
    state_ikfom state_updated = kf.get_x(); //获取当前状态
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Matrix3d rot = eulerToRotationMatrix(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());
    

    //更新状态
    state_updated.pos = pos;
    state_updated.rot = Sophus::SO3(rot);
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
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

    //保存特征点降采样集合
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    updatePath(thisPose6D); //  可视化update后的path
}
//对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
    //注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释

    Eigen::Isometry3d T_b_lidar = Eigen::Isometry3d::Identity();
    //设置旋转部分
    T_b_lidar.linear() = state_point.offset_R_L_I.matrix(); // 提取旋转矩阵并赋值
    // 设置平移部分
    T_b_lidar.translation() = state_point.offset_T_L_I; // 提取平移向量并赋值


    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;        

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
void recontructIKdTree(){
    if(isRecontructKdTree && updateKdtreeCount >  0){
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        //kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap的pose集合
        //降采样
        pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;
        downSizeFilterSubMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
        downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);         //  subMap poses  downsample
        //提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
        }
        // 降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

        std::cout << "subMapKeyFramesDS sizes  =  "   << subMapKeyFramesDS->points.size()  << std::endl;
        
        ikdtree.reconstruct(subMapKeyFramesDS->points);
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed  ikdtree ");
    }
        updateKdtreeCount ++ ; 
}
//更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
void correctPoses()      
{
    if (cloudKeyPoses3D->points.empty())
        return;

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

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);
        }
        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree();
        ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}


//////////////////////////////////////////////////////////////////////////////////
//rviz展示闭环边
void visualizeLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";

    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}
//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
    int loopKeyPre = -1;
    //当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    ROS_INFO("Find loop clousre frame ");
    return true;
}
//提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}
void performLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //时间戳
    string odometryFrame = "camera_init";

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
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
        // 发布闭环匹配关键帧局部map
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
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
        return;
    std::cout << "icp  success  " << std::endl;

    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
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
//回环检测线程
void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok() && startFlag)
    {
        rate.sleep();
        performLoopClosure();   //  回环检测
        visualizeLoopClosure(); // rviz展示闭环边
    }
}
////////////////////////////////////////////////////////////////////////////////////


//信号处理函数，用于捕获退出信号  @param sig 信号编号
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

//将点从雷达坐标系转换到世界坐标系
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

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

/**
 * 同步激光雷达和IMU数据包
 * @param meas 测量组
 * @return 是否成功同步
 */
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();

        if (meas.lidar->points.size() <= 5) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scan_time;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scan_time)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scan_time;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scan_time += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scan_time) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

//更新激光雷达视场中的局部地图范围
void lasermap_fov_segment()
{
    cub_needrm.clear(); //清空需要删除的立方体
    kdtree_delete_counter = 0; //重置k-d树删除计数器

    if (!is_localmap_init){ //局部地图范围初始化
        for (int i = 0; i < 3; i++){
            localmap_range.vertex_min[i] = lidar_position(i) - cube_len / 2.0;//局部地图的最小顶点 1000
            localmap_range.vertex_max[i] = lidar_position(i) + cube_len / 2.0;//局部地图的最大顶点
        }
        is_localmap_init = true;
        return;
    }
    float dist_to_map_edge[3][2];//当前位姿到局部地图边缘的距离，距离边界小于1.5*300，则需要移动边界
    bool is_need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(lidar_position(i) - localmap_range.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(lidar_position(i) - localmap_range.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) 
            is_need_move = true;
    }
    if (!is_need_move) 
        return;
    BoxPointType new_localmap_range, tmp_range;
    new_localmap_range = localmap_range;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_range = localmap_range;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            new_localmap_range.vertex_max[i] -= mov_dist;
            new_localmap_range.vertex_min[i] -= mov_dist;
            tmp_range.vertex_min[i] = localmap_range.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_range);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            new_localmap_range.vertex_max[i] += mov_dist;
            new_localmap_range.vertex_min[i] += mov_dist;
            tmp_range.vertex_max[i] = localmap_range.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_range);
        }
    }
    localmap_range = new_localmap_range;

    //收集点云缓存
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);


    if(cub_needrm.size() > 0) 
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);//删除点

}

//增量更新地图
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_lidar->points[i]), &(feats_down_world->points[i]));
        
        if (!Nearest_Points[i].empty() && is_ekf_init)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool is_need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);//近邻点与当前点距离大，则不需要下采样，直接添加
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[j], mid_point) < dist) //如果近邻点距离 < 当前点距离，不添加该点
                {
                    is_need_add = false;
                    break;
                }
            }
            if (is_need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
}


//发布消息
void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix() * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

int pcd_index = 0;
void publish_frame_world(const ros::Publisher &pubLaserCloudFull_)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_lidar);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i],
                             &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull_.publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&feats_undistort->points[i],
                             &laserCloudWorld->points[i]);
        }

        static int scan_wait_num = 0;
        scan_wait_num++;

        if (scan_wait_num % 4 == 0)
            *pcl_wait_save += *laserCloudWorld;

        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);

}
template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);

    auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];
}
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);

        //save  unoptimized pose
        Eigen::Matrix3d rotationMatrix = state_point.rot.matrix();
        Eigen::Vector3d eulerAngles = rotationMatrix.eulerAngles(2, 1, 0); 

        PointTypePose thisPose6D;  
        thisPose6D.x = msg_body_pose.pose.position.x ;
        thisPose6D.y = msg_body_pose.pose.position.y ;
        thisPose6D.z = msg_body_pose.pose.position.z ;
        thisPose6D.roll = eulerAngles(2) ;
        thisPose6D.pitch = eulerAngles(1) ;
        thisPose6D.yaw = eulerAngles(0) ;
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);   
    }
}

void publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

//发布局部关键帧map的特征点云
void publishGlobalMap()
{
    /*** if path is too large, the rvis will crash ***/
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = "camera_init";
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kdtree查找最近一帧关键帧相邻的关键帧集合
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    //降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // 提取局部相邻关键帧对应的特征点云
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
        // 距离过大
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;

        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
    }
    // 降采样，发布
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}





int main(int argc, char** argv) 
{
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }

    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;

    //降采样过滤器
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization



    //ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01; //重新线性化阈值
    parameters.relinearizeSkip = 1; //跳过重新线性化的次数
    isam = new gtsam::ISAM2(parameters);

    //初始化地图时间戳和帧
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    //IMU处理器参数
    V3D Lidar_T_wrt_IMU(Zero3d);
    M3D Lidar_R_wrt_IMU(Eye3d);
    std::shared_ptr<ImuProcessor> imuProcessor(new ImuProcessor());
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    imuProcessor->set_params(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov),
                      V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);//一秒执行5000次



    

    ros::Subscriber sub_pcl = nh.subscribe("/livox/lidar", 200000, livox_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 200000, imu_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("fast_lio_sam/path_update", 100000);                 //isam更新后的path
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/keyframe_submap", 1);       //发布局部关键帧map的特征点云
    pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/map_global_optimized", 1);  // 发布局部关键帧map的特征点云

 
    //loop closure回环检测
    // 发布闭环匹配关键帧局部map
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // 发布闭环边，rviz中表现为闭环帧之间的连线
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    // 回环检测线程
    std::thread loopthread(&loopClosureThread);

    while(ros::ok())
    {

        if (flg_exit) 
            break;
        ros::spinOnce();//处理一次回调函数

        if(sync_packages(Measures))
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
            lasermap_fov_segment();

            //下采样得到当前帧的点云
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_lidar);
            feats_down_size = feats_down_lidar->points.size();
            
            //当前帧点云数量少，则警告
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //初始化k-d树，存第一帧点云
            if(ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_lidar->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
                continue;

            }

            //迭代状态估计
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_lidar, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            state_point = kf.get_x();
            lidar_position = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;


            getCurPose(state_point); //更新transformTobeMapped
            //back end 后端
            // 1.计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2.添加激光里程计因子、GPS因子、闭环因子
            // 3.执行因子图优化
            // 4.得到当前帧优化后的位姿，位姿协方差
            // 5.添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
            correctPoses();



            publish_odometry(pubOdomAftMapped);

            //向地图k-d树里添加点云
            feats_down_world->resize(feats_down_size);
            map_incremental();

            //发布数据
            if (path_en){
                publish_path(pubPath);
                publish_path_update(pubPathUpdate);  //发布经过isam2优化后的路径
                static int jjj = 0;
                jjj++;
                if (jjj % 100 == 0)
                {
                    publishGlobalMap();              //发布局部点云特征地图
                }
            }
            publish_frame_world(pubLaserCloudFull);
            publish_frame_body(pubLaserCloudFull_body);

            std::cout << "feats_down_size: " << feats_down_size << std::endl;

        }
        rate.sleep();

    }

    return 0;
}