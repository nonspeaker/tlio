#program once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cassert>
#include "common_lib.hpp"

using namespace std;
using namespace Eigen;

//定义点类型
using SCPointType = pcl::PointXYZI; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;

//常量定义
constexpr double LIDAR_HEIGHT = 0.8;
constexpr int PC_NUM_RING = 20;
constexpr int PC_NUM_SECTOR = 60;
constexpr double PC_MAX_RADIUS = 80.0;
constexpr double PC_MAX_ANGLE = 360.0;
constexpr double SC_DIST_THRES = 0.5;
constexpr int NUM_EXCLUDE_RECENT = 50;
constexpr int NUM_CANDIDATES_FROM_TREE = 10;
constexpr int TREE_MAKING_PERIOD = 10;



// 工具函数
inline float xy2Theta(const float &x, const float &y) {
    if (x == 0.0f && y == 0.0f) return 0.0f;
    float angle = atan2(y, x) * (180.0 / M_PI);
    return (angle < 0) ? angle + 360.0f : angle;
}

inline std::vector<float> matrix2Vector(const MatrixXd &matrix) {
    return std::vector<float>(matrix.data(), matrix.data() + matrix.size());
}


class SCManager 
{
public:
    MatrixXd generateScanContext(pcl::PointCloud<SCPointType> &inputCloud);
    MatrixXd computeRingKey(const Eigen::MatrixXd &scan_context);
    MatrixXd computeSectorKey(const Eigen::MatrixXd &scan_context);
    void makeAndSaveScanContextAndKeys(pcl::PointCloud<SCPointType> &input_cloud);
    std::pair<double, int> computeScanContextDistance(const MatrixXd &sc1, const MatrixXd &sc2);
    std::pair<int, float> detectLoopClosure();

private:
    // 私有成员变量
    vector<MatrixXd> scan_contexts;
    vector<MatrixXd> ring_keys;

    vector<vector<float>> ring_key_vectors;
    vector<vector<float>> search_ring_keys;
    std::unique_ptr<InvKeyTree> ring_key_tree;
    int tree_making_period_counter = 0;

    // 私有成员函数
    MatrixXd shiftColumns(const MatrixXd &matrix, int shiftAmount);
    int findBestShift(const MatrixXd &sectorKey1, const MatrixXd &sectorKey2);
    double computeDistance(const MatrixXd &sc1, const MatrixXd &sc2);
};
    
