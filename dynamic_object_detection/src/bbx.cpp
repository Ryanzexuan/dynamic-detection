#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>

class PointCloudProcessor {
public:
    PointCloudProcessor() {
        // 初始化订阅和发布
        pointcloud_sub_ = nh_.subscribe("pointcloud_dynamic", 10, &PointCloudProcessor::pointCloudCallback, this);
        bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/bounding_boxes", 10);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // 将 ROS 点云消息转换为 PCL 点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // 执行 DBSCAN 聚类
        std::vector<pcl::PointIndices> cluster_indices = performDBSCAN(cloud, 0.2, 20);

        // 创建 MarkerArray 来存储多个边界框
        visualization_msgs::MarkerArray boxes;

        // 遍历每个聚类，计算 bounding box 并加入 MarkerArray
        int marker_id = 0;
        for (const auto& indices : cluster_indices) {
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, indices.indices, min_pt, max_pt);

            // 创建一个 Marker 来显示 bounding box
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";  // 使用合适的坐标系
            marker.header.stamp = ros::Time::now();
            marker.ns = "bounding_boxes";
            marker.id = marker_id++;  // 为每个聚类分配一个唯一的 ID
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;

            marker.color.r = 0.00;
            marker.color.g = 1.00;
            marker.color.b = 1.00;
            marker.color.a = 1.00;
            marker.scale.x = 0.1;

            geometry_msgs::Point point[8];
            point[0].x = min_pt[0]; point[0].y = max_pt[1]; point[0].z = max_pt[2];
            point[1].x = min_pt[0]; point[1].y = min_pt[1]; point[1].z = max_pt[2];
            point[2].x = min_pt[0]; point[2].y = min_pt[1]; point[2].z = min_pt[2];
            point[3].x = min_pt[0]; point[3].y = max_pt[1]; point[3].z = min_pt[2];
            point[4].x = max_pt[0]; point[4].y = max_pt[1]; point[4].z = min_pt[2];
            point[5].x = max_pt[0]; point[5].y = min_pt[1]; point[5].z = min_pt[2];
            point[6].x = max_pt[0]; point[6].y = min_pt[1]; point[6].z = max_pt[2];
            point[7].x = max_pt[0]; point[7].y = max_pt[1]; point[7].z = max_pt[2];
            for (int l = 0; l < 8; l++) {
            marker.points.push_back(point[l]);
            }
            marker.points.push_back(point[0]);
            marker.points.push_back(point[3]);
            marker.points.push_back(point[2]);
            marker.points.push_back(point[5]);
            marker.points.push_back(point[6]);
            marker.points.push_back(point[1]);
            marker.points.push_back(point[0]);
            marker.points.push_back(point[7]);
            marker.points.push_back(point[4]);
            boxes.markers.push_back(marker); 

            // 将 marker 添加到 MarkerArray
            boxes.markers.push_back(marker);
        }

        // 发布 MarkerArray
        bbox_pub_.publish(boxes);
    }

private:
    // 执行 DBSCAN 聚类
    std::vector<pcl::PointIndices> performDBSCAN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double eps, int min_pts) {
        std::vector<pcl::PointIndices> cluster_indices;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(eps);  // 距离阈值
        ec.setMinClusterSize(min_pts);  // 最小点数
        ec.setMaxClusterSize(5000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        return cluster_indices;
    }

    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher bbox_pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "bbx");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
