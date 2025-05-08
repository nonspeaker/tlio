#include "local_map.h"
#include <pcl/io/pcd_io.h>


LocalMapManager::LocalMapManager()
{

    mov_threshold = 1.5f;

    det_range = 300.0f;
    cube_len = 200.0f;
    filter_size_map_min = 0.5;


}

LocalMapManager::~LocalMapManager() {}

void LocalMapManager::setParams(float detRange, double filterSizeMapMin, float cubeLen){


    det_range = detRange;
    filter_size_map_min = filterSizeMapMin;
    cube_len = cubeLen;

}

void LocalMapManager::pointLidarToIMU(const PointType &pi, PointType &po, const state_ikfom &state)
{
    Eigen::Vector3d p_lidar(pi.x, pi.y, pi.z);
    Eigen::Vector3d p_IMU(state.offset_R_L_I.matrix() * p_lidar + state.offset_T_L_I);

    po.x = p_IMU(0);
    po.y = p_IMU(1);
    po.z = p_IMU(2);
    po.intensity = pi.intensity;
}
void LocalMapManager::pointLidarToWorld(const PointType &pi, PointType &po, const state_ikfom &state) 
{
    Eigen::Vector3d p_lidar(pi.x, pi.y, pi.z);
    Eigen::Vector3d p_world(state.rot.matrix() * (state.offset_R_L_I.matrix() * p_lidar + state.offset_T_L_I) + state.pos);

    po.x = p_world(0);
    po.y = p_world(1);
    po.z = p_world(2);
    po.intensity = pi.intensity;
}
void LocalMapManager::transformToIMU(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state)
{
    int size = inputCloud->points.size();
    outputCloud->resize(size);
    #pragma omp parallel for
    for(int i = 0; i < size; ++i)
        pointLidarToIMU(inputCloud->points[i], outputCloud->points[i], state);

}
void LocalMapManager::transformToWorld(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, const state_ikfom &state)
{
    int size = inputCloud->points.size();
    outputCloud->resize(size);
    #pragma omp parallel for
    for(int i = 0; i < size; ++i)
        pointLidarToWorld(inputCloud->points[i], outputCloud->points[i], state);

}

void LocalMapManager::savePointCloud(const pcl::PointCloud<PointType>::Ptr& cloud, const std::string& filename) {
    if (cloud->empty()) {
        std::cerr << "Point cloud is empty, cannot save to file: " << filename << std::endl;
        return;
    }

    if (pcl::io::savePCDFileBinary(filename, *cloud) == -1) {
        std::cerr << "Failed to save point cloud to file: " << filename << std::endl;
    } else {
        std::cout << "Point cloud saved to file: " << filename << std::endl;
    }
}

void LocalMapManager::initializeKdTree(KD_TREE &ikdtree, const PointCloudXYZI::Ptr &featsDownLidar, const state_ikfom &state) {

    ikdtree.set_downsample_param(filter_size_map_min);

    PointCloudXYZI::Ptr featsDownWorld(new PointCloudXYZI());//当前帧去畸变后的点云下采样后的点云（世界坐标系）
    
    transformToWorld(featsDownLidar, featsDownWorld, state);

    ikdtree.Build(featsDownWorld->points);
}

void LocalMapManager::updateLocalMapRange(const Eigen::Vector3d &lidarPosition, KD_TREE &ikdtree)
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

void LocalMapManager::updateMapIncremental(const PointCloudXYZI::Ptr &featsDownLidar, KD_TREE &ikdtree, const vector<PointVector> &nearestPoints, const state_ikfom &state, bool &is_ekf_init) {
    PointVector pointToAdd;
    PointVector pointNoNeedDownsample;

    int featsDownSize = featsDownLidar->points.size();
    pointToAdd.reserve(featsDownSize);
    pointNoNeedDownsample.reserve(featsDownSize);

    for (int i = 0; i < featsDownSize; i++) {
        // 转换到世界坐标系
        PointType worldPoint;
        pointLidarToWorld(featsDownLidar->points[i], worldPoint, state);

        if (!nearestPoints[i].empty() && is_ekf_init) {
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
            if (pointsNear.size() >= NUM_MATCH_POINTS)
            {
                for (int j = 0; j < NUM_MATCH_POINTS; j++) 
                {
                    if (calc_dist(pointsNear[j], midPoint) < dist) 
                    {
                        isNeedAdd = false;
                        break;
                     }
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

