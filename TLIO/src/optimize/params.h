#pragma once

#include <ros/ros.h>
#include <string>

struct Config {
    // Surrounding map
    float surroundingkeyframeAddingDistThreshold = 1.0;
    float surroundingkeyframeAddingAngleThreshold = 0.2;
    float surroundingKeyframeDensity = 2.0;

    // Loop closure
    bool loopClosureEnableFlag = true;
    float loopClosureFrequency = 4.0;
    float historyKeyframeSearchRadius = 1.5;
    float historyKeyframeSearchTimeDiff = 30.0;
    int historyKeyframeSearchNum = 20;
    float historyKeyframeFitnessScore = 0.3;

    // Visualization
    float globalMapVisualizationSearchRadius = 1000.0;
    float globalMapVisualizationPoseDensity = 10.0;
    float globalMapVisualizationLeafSize = 1.0;

    // Visualize ikdtree map
    bool visulize_IkdtreeMap = true;
    bool recontructKdTree = true;

    // Export settings
    bool savePCD = true;
    std::string savePCDDirectory = "/Downloads/LOAM/";

    // 加载参数
    void loadParamsFromROS(ros::NodeHandle &nh) {
        nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
        nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);
        nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
        nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);
        nh.param<bool>("recontructKdTree", recontructKdTree, false);

    }
};