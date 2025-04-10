#ifndef POINT_CLOUD_PROCESSOR_H
#define POINT_CLOUD_PROCESSOR_H

#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <ikd-Tree/ikd_Tree.h>
#include "use-ikfom.hpp"
#include "common_lib.hpp"


using namespace std;
using namespace Eigen;

class PointCloudProcessor {
public:
    PointCloudProcessor();
    ~PointCloudProcessor();
    void setParams(int lidarType, int scanLine, double blindZone, float detRange, 
        bool featureEnabled, int pointFilterNum, double filterSizeMapMin, float cubeLen);
    // 点云坐标变换
    void transformToWorld(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state);

    // 点云下采样
    void downsamplePointCloud(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, float leafSize);
    
    //更新局部地图范围
    void updateLocalMapRange(const Eigen::Vector3d &lidarPosition, KD_TREE &ikdtree);

    // 增量更新地图
    void updateMapIncremental(const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, KD_TREE &ikdtree, const vector<PointVector> &nearestPoints, const state_ikfom &state);
    

    // 初始化 k-d 树并存储第一帧点云
    void initializeKdTree(KD_TREE &ikdtree, const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, const state_ikfom &state);

    void process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

 
    

private:
 
    int lidar_type; //雷达类型
    int scan_line; //雷达线数
    double blind; //盲区大小（0.1m）
    float det_range;

    bool feature_enabled; //特征点是否开启
    int point_filter_num; //点云滤波数

    //k-dtree
    double filter_size_map_min;
    float mov_threshold;
    float cube_len;

    pcl::VoxelGrid<PointType> voxelFilter; // 用于下采样的体素滤波器

    bool isLocalMapInit = false;
    BoxPointType localmapRange;
    vector<BoxPointType> cubNeedRm;

    void pointLidarToWorld(const PointType &pi, PointType &po, const state_ikfom &state);
};

PointCloudProcessor::PointCloudProcessor()
{
    lidar_type = 1;
    scan_line = 6;
    blind = 0.1;

    feature_enabled = false;
    point_filter_num = 1;

    mov_threshold = 1.5f;
    det_range = 300.0f;
    cube_len = 200.0f;
    filter_size_map_min = 0.5;

    voxelFilter.setLeafSize(0.5, 0.5, 0.5); // 默认值，可在调用时覆盖
}

PointCloudProcessor::~PointCloudProcessor() {}

void PointCloudProcessor::setParams(int lidarType, int scanLine, double blindZone, float detRange, bool featureEnabled, 
    int pointFilterNum, double filterSizeMapMin, float cubeLen) {

    lidar_type = lidarType;
    scan_line = scanLine;
    blind = blindZone;
    det_range = detRange;

    feature_enabled = featureEnabled;
    point_filter_num = pointFilterNum;

    filter_size_map_min = filterSizeMapMin;
    cube_len = cubeLen;

}


void PointCloudProcessor::process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    PointCloudXYZI pl;
    PointCloudXYZI pl_full;

    //当前帧点云数量
    int plsize = msg->point_num;
    pl_full.resize(plsize);
  
    uint valid_num = 0;
  
    for (uint i = 1; i < plsize; i++)
    {
        if ((msg->points[i].line < scan_line) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            valid_num++;
            if (valid_num % point_filter_num == 0)
            {
                pl_full[i].x = msg->points[i].x;
                pl_full[i].y = msg->points[i].y;
                pl_full[i].z = msg->points[i].z;
                pl_full[i].intensity = msg->points[i].reflectivity;
                pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points, curvature unit: ms

                if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) || (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7) && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
                {
                    pl.push_back(pl_full[i]);
                }
            }
        }
    }
    
    *pcl_out = pl;
}

void PointCloudProcessor::pointLidarToWorld(const PointType &pi, PointType &po, const state_ikfom &state) 
{
    Eigen::Vector3d p_lidar(pi.x, pi.y, pi.z);
    Eigen::Vector3d p_world(state.rot.matrix() * (state.offset_R_L_I.matrix() * p_lidar + state.offset_T_L_I) + state.pos);

    po.x = p_world(0);
    po.y = p_world(1);
    po.z = p_world(2);
    po.intensity = pi.intensity;
}

void PointCloudProcessor::initializeKdTree(KD_TREE &ikdtree, const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, const state_ikfom &state) {
    if (ikdtree.Root_Node == nullptr) {
        ikdtree.set_downsample_param(filter_size_map_min);
        featsDownWorld->resize(featsDownLidar->points.size());

        for (size_t i = 0; i < featsDownLidar->points.size(); ++i) {
            pointLidarToWorld(featsDownLidar->points[i], featsDownWorld->points[i], state);
        }

        ikdtree.Build(featsDownWorld->points);
    }
}

void PointCloudProcessor::downsamplePointCloud(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, float leafSize) 
{
    voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
    voxelFilter.setInputCloud(inputCloud);
    voxelFilter.filter(*outputCloud);
}

void PointCloudProcessor::transformToWorld(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state)
{
    int size = inputCloud->points.size();
    outputCloud->resize(size);
    for(int i = 0; i < size; ++i)
        pointLidarToWorld(inputCloud->points[i], outputCloud->points[i], state);

}

void PointCloudProcessor::updateLocalMapRange(const Eigen::Vector3d &lidarPosition, KD_TREE &ikdtree)
{
    cubNeedRm.clear(); // 清空需要删除的立方体

    if (!isLocalMapInit) { // 局部地图范围初始化
        for (int i = 0; i < 3; i++) {
            localmapRange.vertex_min[i] = lidarPosition(i) - cube_len / 2.0; // 局部地图的最小顶点
            localmapRange.vertex_max[i] = lidarPosition(i) + cube_len / 2.0; // 局部地图的最大顶点
        }
        isLocalMapInit = true;
        return;
    }

    float distToMapEdge[3][2]; // 当前位姿到局部地图边缘的距离
    bool isNeedMove = false;
    for (int i = 0; i < 3; i++) {
        distToMapEdge[i][0] = fabs(lidarPosition(i) - localmapRange.vertex_min[i]);
        distToMapEdge[i][1] = fabs(lidarPosition(i) - localmapRange.vertex_max[i]);
        if (distToMapEdge[i][0] <= mov_threshold * det_range || distToMapEdge[i][1] <= mov_threshold * det_range)
            isNeedMove = true;
    }
    if (!isNeedMove)
        return;

    BoxPointType newLocalMapRange, tmpRange;
    newLocalMapRange = localmapRange;
    float movDist = max((cube_len - 2.0 * mov_threshold * det_range) * 0.5 * 0.9, double(det_range * (mov_threshold - 1)));
    for (int i = 0; i < 3; i++) {
        tmpRange = localmapRange;
        if (distToMapEdge[i][0] <= mov_threshold * det_range) {
            newLocalMapRange.vertex_max[i] -= movDist;
            newLocalMapRange.vertex_min[i] -= movDist;
            tmpRange.vertex_min[i] = localmapRange.vertex_max[i] - movDist;
            cubNeedRm.push_back(tmpRange);
        } else if (distToMapEdge[i][1] <= mov_threshold * det_range) {
            newLocalMapRange.vertex_max[i] += movDist;
            newLocalMapRange.vertex_min[i] += movDist;
            tmpRange.vertex_max[i] = localmapRange.vertex_min[i] + movDist;
            cubNeedRm.push_back(tmpRange);
        }
    }
    localmapRange = newLocalMapRange;

    // 收集点云缓存
    PointVector pointsHistory;
    ikdtree.acquire_removed_points(pointsHistory);

    if (!cubNeedRm.empty())
        int kdtreeDeleteCounter = ikdtree.Delete_Point_Boxes(cubNeedRm); // 删除点
}

void PointCloudProcessor::updateMapIncremental(const PointCloudXYZI::Ptr &featsDownLidar, PointCloudXYZI::Ptr &featsDownWorld, KD_TREE &ikdtree, const vector<PointVector> &nearestPoints, const state_ikfom &state) {
    PointVector pointToAdd;
    PointVector pointNoNeedDownsample;

    int featsDownSize = featsDownLidar->points.size();
    pointToAdd.reserve(featsDownSize);
    pointNoNeedDownsample.reserve(featsDownSize);

    for (int i = 0; i < featsDownSize; i++) {
        // 转换到世界坐标系
        PointType worldPoint;
        pointLidarToWorld(featsDownLidar->points[i], worldPoint, state);
        featsDownWorld->points[i] = worldPoint;

        if (!nearestPoints[i].empty()) {
            const PointVector &pointsNear = nearestPoints[i];
            bool isNeedAdd = true;
            PointType midPoint;
            midPoint.x = floor(worldPoint.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            midPoint.y = floor(worldPoint.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            midPoint.z = floor(worldPoint.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(worldPoint, midPoint);
            if (fabs(pointsNear[0].x - midPoint.x) > 0.5 * filter_size_map_min && fabs(pointsNear[0].y - midPoint.y) > 0.5 * filter_size_map_min && fabs(pointsNear[0].z - midPoint.z) > 0.5 * filter_size_map_min)
            {
                pointNoNeedDownsample.push_back(worldPoint);//近邻点与当前点距离大，则不需要下采样，直接添加
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++) {
                if (pointsNear.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(pointsNear[j], midPoint) < dist) {
                    isNeedAdd = false;
                    break;
                }
            }
            if (isNeedAdd)
                pointToAdd.push_back(worldPoint);
        } else {
            pointToAdd.push_back(worldPoint);
        }
    }

    ikdtree.Add_Points(pointToAdd, true);
    ikdtree.Add_Points(pointNoNeedDownsample, false);
}


#endif // POINT_CLOUD_PROCESSOR_H   