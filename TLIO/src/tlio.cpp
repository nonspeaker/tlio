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

#include "optimization.h"

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

/******************ROS 配置参数 ******************** */
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
double filter_size_surf_min = 1;
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

int pcd_save_interval = -1;

bool timediff_set_flg = false;
double timediff_lidar_to_imu = 0.0;
double last_timestamp_lidar = 0.0;
double last_timestamp_imu = -1.0;



// voxel filter paprams
float odometrySurfLeafSize = 0.4;
float mappingCornerLeafSize = 0.2;
float mappingSurfLeafSize = 0.4;

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

bool visulize_IkdtreeMap = true;
bool recontructKdTree = true;
bool savePCD = true;
bool savePCDDirectory;
/***************************************************** */


mutex mtx_buffer;
condition_variable sig_buffer;
std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<double> time_buffer;


//历史关键帧位置3D（xyz）
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());   
//历史关键帧位姿6D（欧拉角+xyz）         
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); 
//历史关键帧的所有平面点集合(没有降采样)
std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;    

      // 历史关键帧位姿（位置）
// 历史关键帧位姿

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

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());//当前帧去畸变后的点云
PointCloudXYZI::Ptr feats_down_lidar(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云pcd_index
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云（世界坐标系）

std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());
std::shared_ptr<GTSAMOptimizer> gtsamOptimizer(new GTSAMOptimizer(state_point, cloudKeyPoses3D, cloudKeyPoses6D, surfCloudKeyFrames));
std::shared_ptr<LoopClosure> loopClosure(new LoopClosure(state_point, cloudKeyPoses3D, cloudKeyPoses6D, surfCloudKeyFrames));


KD_TREE ikdtree;
int kdtree_delete_counter = 0;
bool is_localmap_init = false;
const float MOV_THRESHOLD = 1.5f;
BoxPointType localmap_range;
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 


nav_msgs::Path path;//lidar移动路径信息的消息
nav_msgs::Path globalPath; //优化后的全局路径
nav_msgs::Odometry odomAftMapped;//建图后里程计的消息
geometry_msgs::PoseStamped msg_body_pose;//机体位姿及其时间戳的消息


bool flg_exit = false;
//信号处理函数，用于捕获退出信号  @param sig 信号编号
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

//将点从雷达坐标系转换到世界坐标系
void pointLidarToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
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
        pointLidarToWorld(&(feats_down_lidar->points[i]), &(feats_down_world->points[i]));
        
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
void publish_frame_world(const ros::Publisher &pubPointCloudWorld)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloud(dense_pub_en ? feats_undistort : feats_down_lidar);
        int size = laserCloud->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointLidarToWorld(&laserCloud->points[i],&laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubPointCloudWorld.publish(laserCloudmsg);
    }

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
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}
void publish_LoopConstraint(const ros::Publisher pubLoopConstraintEdge)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";

    if (loopClosure->loopIndexContainer.empty())
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
    for (auto it = loopClosure->loopIndexContainer.begin(); it != loopClosure->loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = cloudKeyPoses6D->points[key_cur].x;
        p.y = cloudKeyPoses6D->points[key_cur].y;
        p.z = cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = cloudKeyPoses6D->points[key_pre].x;
        p.y = cloudKeyPoses6D->points[key_pre].y;
        p.z = cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}


//闭环约束
ros::Publisher pubLoopConstraintEdge;
//回环检测线程
void loopClosureThread()
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
        publish_LoopConstraint(pubLoopConstraintEdge);
    }
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;

    //初始化地图时间戳和帧
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    //点云降采样
    pcl::VoxelGrid<PointType> downSizeFilterSurf; 
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

    ros::Publisher pubPointCloudWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("fast_lio_sam/path_update", 100000);                 //isam更新后的path
    
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
                    pointLidarToWorld(&(feats_down_lidar->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
                continue;

            }

            //迭代状态估计
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_lidar, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            state_point = kf.get_x();
            lidar_position = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

           //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
            Eigen::Vector3d eulerAngle = state_point.rot.matrix().eulerAngles(2,1,0); 
            gtsamOptimizer->setInitialPose(eulerAngle,state_point.pos, lidar_end_time);
            gtsamOptimizer->optimize(kf, ikdtree, feats_undistort, globalPath, loopClosure);
 


            publish_odometry(pubOdomAftMapped);

            //向地图k-d树里添加点云
            feats_down_world->resize(feats_down_size);
            map_incremental();

            //发布数据
            if (path_en){
                publish_path(pubPath);               //发布原始路径
                //publish_path_update(pubPathUpdate);  //发布经过isam2优化后的路径
            }
            publish_frame_world(pubPointCloudWorld); //发布全局点云


            std::cout << "feats_down_size: " << feats_down_size << std::endl;

        }
        rate.sleep();

    }

    return 0;
}