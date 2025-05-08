#pragma once


#include "params.h"
#include "loopclosure/common.h"


class OptimizationManager {
    public:
        OptimizationManager(float distThresh, float angleThresh, float fitnessScore,
                pcl::PointCloud<PointType>::Ptr &keyPos3D, pcl::PointCloud<PointTypePose>::Ptr &keyPos6D, std::vector<PointCloudXYZI::Ptr> &keyClouds)
            : surroundingKeyframeAddingDistThreshold(distThresh), surroundingKeyframeAddingAngleThreshold(angleThresh),
            historyKeyframeFitnessScore(fitnessScore), keyFramePositions3D(keyPos3D), keyFramePoses6D(keyPos6D), keyCloudVector(keyClouds) 
        {
            // 初始化 GTSAM 优化器
            gtsam::ISAM2Params parameters;
            parameters.relinearizeThreshold = 0.01;
            parameters.relinearizeSkip = 1;
            isam = gtsam::ISAM2(parameters);

            // 初始化噪声模型
            odometryNoise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

            priorNoise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());

            double loopNoiseScore = 1e-1;
            gtsam::Vector robustNoiseVector6(6);
            robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
                loopNoiseScore, loopNoiseScore, loopNoiseScore;

            robustLoopNoise = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
        }


        bool isKeyFrame(const PointTypePose &current_pose) {
            if (keyFramePositions3D->points.empty())
                return true;
    
            Eigen::Affine3f transStart = pclPointToAffine3f(keyFramePoses6D->back());
            Eigen::Affine3f transFinal = pclPointToAffine3f(current_pose);
    
            Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
    
            return !(abs(roll) < surroundingKeyframeAddingAngleThreshold &&
                     abs(pitch) < surroundingKeyframeAddingAngleThreshold &&
                     abs(yaw) < surroundingKeyframeAddingAngleThreshold &&
                     sqrt(x * x + y * y + z * z) < surroundingKeyframeAddingDistThreshold);
        }
    
        bool icpAlign(const PointCloudXYZI::Ptr& src, const PointCloudXYZI::Ptr& tgt, Eigen::Matrix4f& out_transform) {
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
                return true;
            }
        }

        void addOdomFactor(const PointTypePose &current_pose) {

            int cloudInd = keyFramePositions3D->size() - 1;
        
            if (!cloudInd) 
            {
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                    0, pclPointTogtsamPose3(current_pose), priorNoise));
                initial.insert(0, pclPointTogtsamPose3(current_pose));
            } 
            else 
            {
                auto prev_pose = pclPointTogtsamPose3(keyFramePoses6D->points[cloudInd - 1]);
                auto curr_pose = pclPointTogtsamPose3(current_pose);
                graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    cloudInd - 1, cloudInd, prev_pose.between(curr_pose),
                    odometryNoise));
    
                initial.insert(cloudInd, pclPointTogtsamPose3(current_pose));
            }
        }

        void addLoopFactor(int loopKeyCur, int loopKeyPre, const Eigen::Matrix4f& correction) {
            // 当前关键帧的原始位姿
            Eigen::Affine3f tWrong = pclPointToAffine3f(keyFramePoses6D->points[loopKeyCur]);
            // 校正后的位姿
            Eigen::Affine3f tCorrect = Eigen::Affine3f(correction) * tWrong;
        
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(keyFramePoses6D->points[loopKeyPre]);
        
            // 添加回环因子
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                loopKeyCur, loopKeyPre, poseFrom.between(poseTo), robustLoopNoise));
        }

        void optimize(bool has_loop_flag) 
        {

            isam.update(graph, initial);
            isam.update();
            // 如果有回环，进行多次更新以提高收敛性
            if (has_loop_flag) {
                isam.update();
                isam.update();
                isam.update();
                isam.update();
                isam.update();
            }

            graph.resize(0);
            initial.clear();

            gtsam::Values isamCurrentEstimate = isam.calculateEstimate();
            for (int i = 0; i < isamCurrentEstimate.size(); ++i) {
                keyFramePositions3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
                keyFramePositions3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
                keyFramePositions3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();
    
                keyFramePoses6D->points[i].x = keyFramePositions3D->points[i].x;
                keyFramePoses6D->points[i].y = keyFramePositions3D->points[i].y;
                keyFramePoses6D->points[i].z = keyFramePositions3D->points[i].z;
                keyFramePoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
                keyFramePoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
                keyFramePoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();
            }

        }

    private:
        float surroundingKeyframeAddingDistThreshold;
        float surroundingKeyframeAddingAngleThreshold;
        float historyKeyframeFitnessScore;

        pcl::PointCloud<PointType>::Ptr& keyFramePositions3D;
        pcl::PointCloud<PointTypePose>::Ptr& keyFramePoses6D;
        std::vector<PointCloudXYZI::Ptr>& keyCloudVector;
    
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;
        gtsam::ISAM2 isam;

        // 噪声模型
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise;
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise;
        gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
};

