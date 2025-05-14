#pragma once
#include "../common.h"
#include <visualization_msgs/MarkerArray.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <thread>
#include <atomic>
#include <chrono>

class RadiusManager {

public:
    RadiusManager(float searchRadius, float timeDiff, int searchNum, float fitnessScore,
                  pcl::PointCloud<PointType>::Ptr &keyFramePositions,
                  pcl::PointCloud<PointTypePose>::Ptr &keyFramePoses,
                  std::vector<PointCloudXYZI::Ptr> &keyCloudVec)
          : keyFramePositions3D(keyFramePositions), keyFramePoses6D(keyFramePoses), keyCloudVector(keyCloudVec),
            historyKeyframeFitnessScore(fitnessScore), loopThreadRunning(false) 
    {
        historyKeyframeSearchRadius = searchRadius;
        historyKeyframeSearchTimeDiff = timeDiff;
        historyKeyframeSearchNum = searchNum;
        loopIndexContainer.clear();
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    }

    ~RadiusManager() {
        stopLoopDetectionThread(); // 确保线程在析构时停止
    }

    void startLoopDetectionThread(std::vector<std::pair<int, int>>& loopIndexQueue,
        std::vector<gtsam::Pose3>& loopPoseQueue,
        std::vector<gtsam::noiseModel::Diagonal::shared_ptr>& loopNoiseQueue,
        ros::Publisher& pubLoopMarkers) 
    {
        loopThreadRunning = true;
        loopThread = std::thread(&RadiusManager::loopDetectionThread, this,
        std::ref(loopIndexQueue),
        std::ref(loopPoseQueue),
        std::ref(loopNoiseQueue),
        std::ref(pubLoopMarkers));
    }
    // 停止回环检测线程
    void stopLoopDetectionThread() {
        loopThreadRunning = false;
        if (loopThread.joinable()) {
            loopThread.join();
            }      
        }

    // 检测回环并输出用于 ICP 的点云
    bool detectLoopAndGetICPClouds(int loopKeyCur, int &loopKeyPre, PointCloudXYZI::Ptr &nearKeyframeCloud) {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        // 如果已经检测到回环，则不再进行检测
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

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

    // ICP 对齐函数
    bool icpAlign(const PointCloudXYZI::Ptr &src, const PointCloudXYZI::Ptr &tgt, Eigen::Matrix4f &out_transform, float &noiseScore) {
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(src);
        icp.setInputTarget(tgt);

        PointCloudXYZI::Ptr result(new PointCloudXYZI());
        icp.align(*result);

        if (!icp.hasConverged() || icp.getFitnessScore() > historyKeyframeFitnessScore) {
            return false;
        } else {
            out_transform = icp.getFinalTransformation();
            noiseScore = icp.getFitnessScore();
            return true;
        }
    }

private:
    void publishLoopMarkers(ros::Publisher& pubLoopMarkers) {
        visualization_msgs::MarkerArray markerArray;

        // 闭环顶点
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
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
        markerEdge.header.frame_id = "camera_init";
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
        {
            std::lock_guard<std::mutex> lock(loopMutex); // 确保线程安全
            for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) {
                int key_cur = it->first;
                int key_pre = it->second;

                geometry_msgs::Point p;
                p.x = keyFramePoses6D->points[key_cur].x;
                p.y = keyFramePoses6D->points[key_cur].y;
                p.z = keyFramePoses6D->points[key_cur].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);

                p.x = keyFramePoses6D->points[key_pre].x;
                p.y = keyFramePoses6D->points[key_pre].y;
                p.z = keyFramePoses6D->points[key_pre].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
            }
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);

        pubLoopMarkers.publish(markerArray);
    }


    void loopDetectionThread(std::vector<std::pair<int, int>>& loopIndexQueue, std::vector<gtsam::Pose3>& loopPoseQueue,
        std::vector<gtsam::noiseModel::Diagonal::shared_ptr>& loopNoiseQueue, ros::Publisher& pubLoopMarkers) 
    {
        while (loopThreadRunning) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 控制检测频率

            if (keyFramePoses6D->empty()) {
            continue; // 如果没有关键帧，跳过检测
            }

            int loopKeyCur = keyFramePoses6D->size() - 1; // 当前关键帧索引
            int loopKeyPre = -1; // 候选闭环匹配帧索引
            PointCloudXYZI::Ptr nearKeyframeCloud(new PointCloudXYZI()); // 历史关键帧子地图点云

            // 检测回环并提取历史关键帧点云
            if (!detectLoopAndGetICPClouds(loopKeyCur, loopKeyPre, nearKeyframeCloud)) {
            continue; // 如果未检测到回环，跳过
            }

            // 提取当前关键帧点云
            PointCloudXYZI::Ptr currentKeyframeCloud(new PointCloudXYZI());
            currentKeyframeCloud = transformPointCloud(keyCloudVector[loopKeyCur], keyFramePoses6D->points[loopKeyCur]);

            // 执行 ICP 对齐
            Eigen::Matrix4f correction;
            float noiseScore = 0.0f;
            if (!icpAlign(currentKeyframeCloud, nearKeyframeCloud, correction, noiseScore)) {
            std::cout << "ICP alignment failed for loop closure." << std::endl;
            continue; // 如果 ICP 对齐失败，跳过
            }

            std::cout << "ICP alignment succeeded for loop closure." << std::endl;

            // 计算优化后的位姿
            Eigen::Affine3f correctionLidarFrame(correction);
            Eigen::Affine3f tWrong = pclPointToAffine3f(keyFramePoses6D->points[loopKeyCur]); // 当前帧位姿
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // 优化后的位姿

            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);

            gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(keyFramePoses6D->points[loopKeyPre]);

            gtsam::Vector Vector6(6);
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

            // 更新外部队列
            {
            std::lock_guard<std::mutex> lock(loopMutex); // 确保线程安全
            loopIndexQueue.push_back(std::make_pair(loopKeyCur, loopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            }
            loopIndexContainer[loopKeyCur] = loopKeyPre; // 更新闭环索引容器


            publishLoopMarkers(pubLoopMarkers); // 发布闭环标记


        }
    }

    
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    float historyKeyframeFitnessScore;
    int historyKeyframeSearchNum;


    std::map<int, int> loopIndexContainer;      //闭环索引容器
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr &keyFramePositions3D;
    pcl::PointCloud<PointTypePose>::Ptr &keyFramePoses6D;
    std::vector<PointCloudXYZI::Ptr> &keyCloudVector;

    std::mutex loopMutex; 
    std::thread loopThread; // 回环检测线程
    std::atomic<bool> loopThreadRunning; // 控制线程运行状态
};