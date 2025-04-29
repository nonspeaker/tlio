#include "pcl_process.h"


PointCloudProcessor::PointCloudProcessor()
{
    given_offset_time = false; 

    lidar_type = 1;
    scan_line = 6;
    scan_rate = 10;
    time_unit = 2; // 0: sec, 1: ms, 2: us, 3: ns
    blind = 0.1;

    feature_enabled = false;
    point_filter_num = 1;

    voxelFilter.setLeafSize(0.5, 0.5, 0.5); // 默认值，可在调用时覆盖
}

PointCloudProcessor::~PointCloudProcessor() {}


void PointCloudProcessor::setParams(int lidarType, int scanLine, int scanRate, int timeUnit, double blindZone, bool featureEnabled, int pointFilterNum){
            

    lidar_type = lidarType;
    scan_line = scanLine;
    scan_rate = scanRate;
    time_unit = timeUnit;
    blind = blindZone;

    feature_enabled = featureEnabled;
    point_filter_num = pointFilterNum;
}


void PointCloudProcessor::avia_handler(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
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



void PointCloudProcessor::process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    avia_handler(msg, pcl_out);
}

void PointCloudProcessor::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    PointCloudXYZI pl_surf;
    pl_surf.clear();
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0)
        return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * scan_rate; // scan angular velocity
    std::vector<bool> is_first(scan_line, true);
    std::vector<double> yaw_fp(scan_line, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(scan_line, 0.0);  // yaw of last scan point
    std::vector<float> time_last(scan_line, 0.0); // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
        given_offset_time = true;
    }
    else
    {
        given_offset_time = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--)
        {
        if (pl_orig.points[i].ring == layer_first)
        {
            yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
            break;
        }
        }
    }


    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // curvature unit: ms // cout<<added_pt.curvature<<endl;
        // std::cout << "added_pt.curvature:" << added_pt.curvature << std::endl;


        if (!given_offset_time)
        {
        int layer = pl_orig.points[i].ring;
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

        if (is_first[layer])
        {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
            yaw_fp[layer] = yaw_angle;
            is_first[layer] = false;
            added_pt.curvature = 0.0;
            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
            continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer])
        {
            added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        }
        else
        {
            added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
            added_pt.curvature += 360.0 / omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
        if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
        {
            pl_surf.points.push_back(added_pt);
        }
        }
    }
    *pcl_out = pl_surf;
  
}

void PointCloudProcessor::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    PointCloudXYZI pl_surf;
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    pl_surf.reserve(plsize);
    //#pragma omp parallel for
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
        if (i % point_filter_num != 0)
        continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind * blind))
        continue;

        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

        pl_surf.points.push_back(added_pt);
    }
    *pcl_out = pl_surf;
}

void PointCloudProcessor::rs_handler(const sensor_msgs::PointCloud2_<allocator<void>>::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    PointCloudXYZI pl_surf;
    pl_surf.clear();

    pcl::PointCloud<rslidar_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * scan_rate; // scan angular velocity
    std::vector<bool> is_first(scan_line, true);
    std::vector<double> yaw_fp(scan_line, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(scan_line, 0.0);  // yaw of last scan point
    std::vector<float> time_last(scan_line, 0.0); // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].timestamp > 0) // todo check pl_orig.points[plsize - 1].time
    {
    given_offset_time = true;
    // std::cout << "given_offset_time = true " << std::endl;
    }
    else
    {
    given_offset_time = false;
    double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578; // 记录第一个点(index 0)的yaw， to degree
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring; // 第一个点(index 0)的layer序号
    for (uint i = plsize - 1; i > 0; i--)     // 倒序遍历，找到与第一个点相同layer的最后一个点
    {
        if (pl_orig.points[i].ring == layer_first)
        {
        yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578; // 与第一个点相同layer的最后一个点的yaw
        break;
        }
    }
    }

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = (pl_orig.points[i].timestamp - pl_orig.points[0].timestamp) * 1000.0; // curvature unit: ms
        // std::cout << "added_pt.curvature:" << added_pt.curvature << std::endl;

        if (!given_offset_time)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])
            {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
            yaw_fp[layer] = yaw_angle;
            is_first[layer] = false;
            added_pt.curvature = 0.0;
            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
            continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
            added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
            added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])
            added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind) )
            {
            pl_surf.points.push_back(added_pt);
            }
        }
    }
    *pcl_out = pl_surf;
}


void PointCloudProcessor::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    switch (time_unit)
    {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
    }

    switch (lidar_type)
    {
    case VELO16:
      velodyne_handler(msg, pcl_out);
      break;

    case OUST64:
      oust64_handler(msg, pcl_out);
      break;

    case RS32:
      rs_handler(msg, pcl_out);
      break;
  
    default:
      printf("Error LiDAR Type");
      break;
    }    

}

void PointCloudProcessor::downsamplePointCloud(const PointCloudXYZI::Ptr &inputCloud, PointCloudXYZI::Ptr &outputCloud, float leafSize) 
{
    voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
    voxelFilter.setInputCloud(inputCloud);
    voxelFilter.filter(*outputCloud);
}
