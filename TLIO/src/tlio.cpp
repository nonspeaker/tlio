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

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <livox_ros_driver2/CustomMsg.h>


#include <condition_variable>

#include "imu_process.h"
#include "pcl_process.h"


#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

using namespace std;

int NUM_MAX_ITERATIONS = 4;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
double gyr_cov = 0.1;
double acc_cov = 0.1;
double b_gyr_cov = 0.0001;
double b_acc_cov = 0.0001;

vector<double> extrinT = vector<double>{-0.011, -0.02329, 0.04412};
vector<double> extrinR = vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};

double filter_size_surf_min = 0.5;
double filter_size_map_min = 0.5;


mutex mtx_buffer;
condition_variable sig_buffer;


double time_offset_lidar_to_imu = 0.0;//lidar相对于imu时间偏移，配置参数

std::shared_ptr<PointCloudProcessor> pclProcessor(new PointCloudProcessor());

bool timediff_set_flg = false;
double timediff_lidar_to_imu = 0.0;
double last_timestamp_lidar = 0.0;
double last_timestamp_imu = -1.0;

std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<double> time_buffer;


ros::Publisher pub_pcl;
ros::Publisher pub_imu;



bool lidar_pushed;
double lidar_end_time = 0.0;
double lidar_mean_scan_time = 0.1; //用于同步


bool is_first_scan = true;
bool is_ekf_init = false;
double first_lidar_time = 0.0;

MeasureGroup Measures;//当前雷达测量数据包
esekfom::esekf kf;//卡尔曼滤波器状态
state_ikfom state_point;//当前状态
Eigen::Vector3d lidar_position;//当前雷达位置


PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());//地图中的特征点点云
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());//当前帧去畸变后的点云
PointCloudXYZI::Ptr feats_down_lidar(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云（世界坐标系）
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));//存储法向量
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));//存储原始点云
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));//校正后的法向量
PointCloudXYZI::Ptr _featsArray;


int feats_down_size = 0;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
KD_TREE<PointType> ikdtree;

V3D cur_euler;


double cube_len = 0.0;

nav_msgs::Path path;//lidar移动路径信息的消息
nav_msgs::Odometry odomAftMapped;//建图后里程计的消息
geometry_msgs::Quaternion geoQuat;//集合四元数消息，表示旋转
geometry_msgs::PoseStamped msg_body_pose;//机体位姿及其时间戳的消息

vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 

int kdtree_delete_counter = 0;
int kdtree_delete_time = 0.0;

bool is_localmap_init = false;
BoxPointType localmap_range;

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

bool flg_exit = false;

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
bool scan_pub_en = true, dense_pub_en = true, scan_body_pub_en = true;
bool pcd_save_en = true, time_sync_en = false, extrinsic_est_en = true, path_en = true;
int pcd_save_interval = -1 ,pcd_index = 0;

/**
 * 信号处理函数，用于捕获退出信号
 * @param sig 信号编号
 */
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



void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix() * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}


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
int main(int argc, char** argv) 
{
    ros::init(argc, argv, "tlio");
    ros::NodeHandle nh;
    /*
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);            // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);          // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);                   // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path", map_file_path, "");                    // 地图保存路径
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");         // 雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");           // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);              // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5); // VoxelGrid降采样时的体素大小
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);    // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);               // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);               // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);        // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);        // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);        // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA); // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);       // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);           // 采样间隔，即每隔point_filter_num个点取1个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // 是否将点云地图保存到PCD文件
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R

    cout << "Lidar_type: " << p_pre->lidar_type << endl;
    */
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    ros::Subscriber sub_pcl = nh.subscribe("/livox/lidar", 200000, livox_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 200000, imu_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    std::shared_ptr<ImuProcessor> imuProcessor(new ImuProcessor());
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    imuProcessor->set_params(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov),
                      V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);//一秒执行5000次

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
                imuProcessor->first_lidar_time = first_lidar_time;
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
            is_ekf_init = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;//判断EKF是否初始化，根据当前雷达数据包的时间与第一帧雷达数据包的时间戳的差值是否小于初始化时间
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

            publish_odometry(pubOdomAftMapped);

            //向地图k-d树里添加点云
            feats_down_world->resize(feats_down_size);
            map_incremental();

            //发布数据

            publish_path(pubPath);
            publish_frame_world(pubLaserCloudFull);
            publish_frame_body(pubLaserCloudFull_body);

            std::cout << "feats_down_size: " << feats_down_size << std::endl;

        }
        rate.sleep();

    }

    return 0;
}