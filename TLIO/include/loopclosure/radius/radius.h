#pragma once
#include "../common.h"


#include <pcl/kdtree/kdtree_flann.h>



class RadiusManager {

public:
    RadiusManager(float searchRadius, float timeDiff, int searchNum,
                  pcl::PointCloud<PointType>::Ptr &keyFramePositions,
                  pcl::PointCloud<PointTypePose>::Ptr &keyFramePoses,
                  std::vector<PointCloudXYZI::Ptr> &keyCloudVec)
          :keyFramePositions3D(keyFramePositions), keyFramePoses6D(keyFramePoses), keyCloudVector(keyCloudVec) 
    {
        historyKeyframeSearchRadius=searchRadius,
        historyKeyframeSearchTimeDiff=timeDiff,
        historyKeyframeSearchNum=searchNum,
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    }

    // 检测回环并输出用于 ICP 的点云
    bool detectLoopAndGetICPClouds(int loopKeyCur, int &loopKeyPre, PointCloudXYZI::Ptr &nearKeyframeCloud) {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        // 使用 KD-Tree 搜索历史关键帧
        kdtreeHistoryKeyPoses->setInputCloud(keyFramePositions3D);
        kdtreeHistoryKeyPoses->radiusSearch(
            keyFramePositions3D->points[loopKeyCur], historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
            int id = pointSearchIndLoop[i];
            if (abs(keyFramePoses6D->points[id].time - keyFramePoses6D->points[loopKeyCur].time) > historyKeyframeSearchTimeDiff) {
                loopKeyPre = id;
                // 构建局部子地图点云
                nearKeyframeCloud->clear();
                for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j) {
                    int keyNear = loopKeyPre + j;
                    if (keyNear < 0 || keyNear >= keyCloudVector.size()) {
                        continue;
                    }
                    *nearKeyframeCloud += *transformPointCloud(keyCloudVector[keyNear], keyFramePoses6D->points[keyNear]);
                }
                return true;
            }
        }
        return false;
    }

private:
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int historyKeyframeSearchNum;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr& keyFramePositions3D;
    pcl::PointCloud<PointTypePose>::Ptr& keyFramePoses6D;
    std::vector<PointCloudXYZI::Ptr>& keyCloudVector;
};