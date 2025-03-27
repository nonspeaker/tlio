#include "pcl_process.h"


PointCloudProcessor::PointCloudProcessor()
{
    N_SCANS = 6;
    blind = 0.1;

    point_filter_num = 1;

}

PointCloudProcessor::~PointCloudProcessor() {}


void PointCloudProcessor::process (const livox_ros_driver2::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    PointCloudXYZI pl;
    PointCloudXYZI pl_full;

    //当前帧点云数量
    int plsize = msg->point_num;
    pl_full.resize(plsize);
  
    uint valid_num = 0;
  
    for (uint i = 1; i < plsize; i++)
    {
        if ((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
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