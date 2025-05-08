#include "scancontext.h"

MatrixXd SCManager::shiftColumns(const MatrixXd &matrix, int shiftAmount) {
    assert(shiftAmount >= 0);
    if (shiftAmount == 0) return matrix;

    // 使用 Eigen 的 block 操作避免创建新矩阵
    int totalColumns = matrix.cols();
    MatrixXd shiftedMatrix(matrix.rows(), totalColumns);
    shiftedMatrix << matrix.rightCols(totalColumns - shiftAmount), matrix.leftCols(shiftAmount);
    return shiftedMatrix;
}

int SCManager::findBestShift(const MatrixXd &sectorKey1, const MatrixXd &sectorKey2) {
    int bestShift = 0;
    double minDifferenceNorm = std::numeric_limits<double>::max();

    for (int shift = 0; shift < sectorKey1.cols(); ++shift) {
        MatrixXd shiftedSectorKey2 = shiftColumns(sectorKey2, shift);
        double differenceNorm = (sectorKey1 - shiftedSectorKey2).norm();
        if (differenceNorm < minDifferenceNorm) {
            bestShift = shift;
            minDifferenceNorm = differenceNorm;
        }
    }
    return bestShift;
}

double SCManager::computeDistance(const MatrixXd &sc1, const MatrixXd &sc2) {
    int validColumns = 0;
    double similaritySum = 0.0;

    for (int i = 0; i < sc1.cols(); ++i) {
        const auto &col1 = sc1.col(i);
        const auto &col2 = sc2.col(i);

        if (col1.norm() == 0.0 && col2.norm() == 0.0) continue;

        similaritySum += col1.dot(col2) / (col1.norm() * col2.norm());
        ++validColumns;
    }

    return (validColumns == 0) ? 1.0 : 1.0 - (similaritySum / validColumns);
}


//生成ScanContext描述子
MatrixXd SCManager::generateScanContext(pcl::PointCloud<SCPointType> &inputCloud)
{
    const int NO_POINT = -1000;
    MatrixXd scanContext = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    for (const auto &point : inputCloud.points)
    {
        float adjustedZ = point.z + LIDAR_HEIGHT;
        float radius = sqrt(point.x * point.x + point.y * point.y);
        float angle = xy2Theta(point.x, point.y);

        if (radius > PC_MAX_RADIUS)
            continue;

        int ringIdx = std::clamp(int(ceil((radius / PC_MAX_RADIUS) * PC_NUM_RING)), 1, PC_NUM_RING);
        int sectorIdx = std::clamp(int(ceil((angle / PC_MAX_ANGLE) * PC_NUM_SECTOR)), 1, PC_NUM_SECTOR);

        scanContext(ringIdx - 1, sectorIdx - 1) = std::max(scanContext(ringIdx - 1, sectorIdx - 1), adjustedZ);
    }

    scanContext = (scanContext.array() == NO_POINT).select(0, scanContext);
    return scanContext;
}

// 计算每一行的均值，生成环键
MatrixXd SCManager::computeRingKey(const Eigen::MatrixXd &scan_context)
{
    return scan_context.rowwise().mean();
}

// 计算每一列的均值，生成扇区键
MatrixXd SCManager::computeSectorKey(const Eigen::MatrixXd &scan_context)
{
    return scan_context.colwise().mean();
}

void SCManager::makeAndSaveScanContextAndKeys(pcl::PointCloud<SCPointType> &input_cloud)
{
    // 生成扫描上下文和对应的键
    Eigen::MatrixXd scan_context = makeScanContext(input_cloud);
    Eigen::MatrixXd ring_key = computeRingKey(scan_context);
    std::vector<float> ring_key_vector = matrix2Vector(ring_key);

    // 保存到数据存储中
    scan_contexts.push_back(scan_context);
    ring_keys.push_back(ring_key);
    ring_key_vectors.push_back(ring_key_vector);
}



// 计算两个Scan Context之间的距离，并返回最小距离和最佳对齐的列偏移量
std::pair<double, int> SCManager::computeScanContextDistance(const MatrixXd &sc1, const MatrixXd &sc2)
{
    // 计算两个Scan Context的扇区键
    MatrixXd vkey1 = computeSectorKey(sc1);
    MatrixXd vkey2 = computeSectorKey(sc2);

    // 找到最佳列偏移量，使两个扇区键的差异最小
    int bestShift = findBestShift(vkey1, vkey2);

    // 定义搜索半径，基于SEARCH_RATIO（假设已定义）
    const int searchRadius = round(0.5 * SEARCH_RATIO * sc1.cols());

    // 构建搜索空间，包括最佳偏移量及其附近的偏移量
    std::vector<int> shiftSearchSpace{bestShift};
    for (int i = 1; i <= searchRadius; ++i)
    {
        shiftSearchSpace.push_back((bestShift + i + sc1.cols()) % sc1.cols());
        shiftSearchSpace.push_back((bestShift - i + sc1.cols()) % sc1.cols());
    }

    // 去重并排序搜索空间
    std::sort(shiftSearchSpace.begin(), shiftSearchSpace.end());

    // 初始化最小距离和最佳偏移量
    double minDistance = std::numeric_limits<double>::max();
    int optimalShift = 0;

    // 遍历搜索空间，计算每种偏移量下的距离
    for (int shift : shiftSearchSpace)
    {
        // 将sc2按当前偏移量进行列循环移位
        MatrixXd shiftedSc2 = shiftColumns(sc2, shift);

        // 计算sc1和移位后的sc2之间的距离
        double distance = computeDistance(sc1, shiftedSc2);

        // 更新最小距离和最佳偏移量
        if (distance < minDistance)
        {
            minDistance = distance;
            optimalShift = shift;
        }
    }

    // 返回最小距离和最佳偏移量
    return {minDistance, optimalShift};
}

// 检测回环闭合，返回回环帧ID和对应的偏航角差
std::pair<int, float> SCManager::detectLoopClosure()
{
    int loop_id = -1; // 初始化为 -1，表示未检测到回环

    // 如果数据不足以进行回环检测，直接返回
    if (ring_key_vectors.size() < NUM_EXCLUDE_RECENT + 1)
    {
        return {loop_id, 0.0};
    }

    // 获取当前帧的环键和Scan Context
    const auto &current_ring_key = ring_key_vectors.back();
    const auto &current_scan_context = scan_contexts.back();

    // 定期重建KD树
    if (tree_making_period_counter % TREE_MAKING_PERIOD == 0)
    {
        // 更新搜索用的环键集合，排除最近的NUM_EXCLUDE_RECENT帧
        search_ring_keys.clear();
        search_ring_keys.assign(ring_key_vectors.begin(), ring_key_vectors.end() - NUM_EXCLUDE_RECENT);

        // 重置并重新构建KD树
        ring_key_tree.reset();
        ring_key_tree = std::make_unique<InvKeyTree>(PC_NUM_RING, search_ring_keys, 10);
    }
    tree_making_period_counter++;

    // 使用KD树进行KNN搜索，找到候选帧
    std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

    nanoflann::KNNResultSet<float> knn_result(NUM_CANDIDATES_FROM_TREE);
    knn_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    ring_key_tree->index->findNeighbors(knn_result, &current_ring_key[0], nanoflann::SearchParams(10));

    // 遍历候选帧，找到最优匹配
    double min_distance = std::numeric_limits<double>::max();
    int best_alignment = 0;
    int best_index = -1;

    for (size_t idx : candidate_indexes)
    {
        const auto &candidate_context = scan_contexts[idx];

        // 计算当前帧与候选帧的Scan Context距离
        auto [distance, alignment] = computeScanContextDistance(current_scan_context, candidate_context);

        // 更新最优匹配
        if (distance < min_distance)
        {
            min_distance = distance;
            best_alignment = alignment;
            best_index = idx;
        }
    }

    // 检查是否满足回环条件
    if (min_distance < SC_DIST_THRES)
    {
        loop_id = best_index;
        float yaw_diff_deg = best_alignment * PC_UNIT_SECTORANGLE; // 假设PC_UNIT_SECTORANGLE已定义
        std::cout << "[Loop found] Nearest distance: " << min_distance
                  << " between " << scan_contexts.size() - 1 << " and " << best_index << ".\n";
        std::cout << "[Loop found] Yaw difference: " << yaw_diff_deg << " degrees.\n";
        return {loop_id, deg2rad(yaw_diff_deg)}; // 假设deg2rad函数已定义
    }
    else
    {
        std::cout << "[No loop] Nearest distance: " << min_distance
                  << " between " << scan_contexts.size() - 1 << " and " << best_index << ".\n";
        return {loop_id, 0.0};
    }
}