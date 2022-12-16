#pragma once

#include <vector>

#include "opencv4/opencv2/opencv.hpp"


struct FrameDataContainer
{
    cv::Mat                             image;
    std::vector<cv::KeyPoint>           keypoints_l;
    std::vector<cv::KeyPoint>           keypoints_r;
    std::vector<cv::Mat>                descriptors_l;
    std::vector<cv::Mat>                descriptors_r;
    std::vector<cv::Mat>                points;
    std::vector<int>                    point_global_idx;

    void Flush()
    {
        keypoints_l.clear();
        keypoints_r.clear();
        descriptors_l.clear();
        descriptors_r.clear();
        points.clear();
        point_global_idx.clear();
    }

    void Set(const cv::Mat& img) { image = img; }

    void AddKeyPoints(const cv::KeyPoint& kp_l, const cv::KeyPoint& kp_r)
    {
        keypoints_l.emplace_back(kp_l);
        keypoints_r.emplace_back(kp_r);
    }

    // void AddKeyPoint(const cv::KeyPoint& keypoint) { keypoints.emplace_back(keypoint); }
    void AddDescriptors(const cv::Mat& descriptor_l, const cv::Mat& descriptor_r)
    {
        descriptors_l.emplace_back(descriptor_l);
        descriptors_r.emplace_back(descriptor_r);
    }

    void AddPoint(cv::Mat point) { points.emplace_back(point); }
    void AddGlobalIndex(int idx) { point_global_idx.emplace_back(idx); }
};
