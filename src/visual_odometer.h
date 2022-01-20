#pragma once

#include "camera_model.h"
#include "absolute_orientation.h"

#include <Eigen/Geometry>
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/features2d.hpp"


class VisualOdometer
{
public:
    VisualOdometer();
    ~VisualOdometer();

    void Camera(CameraModel::Stereo* camera);
    CameraModel::Stereo* Camera();

    Eigen::Matrix4f Track(const cv::Mat& img_l, const cv::Mat& img_r);

private:
    // detect and match features across stereo images
    void MatchFeaturesBetweenStereoImages(
        const cv::Mat& img_1, 
        const cv::Mat& img_2, 
        std::vector<cv::Point2f>& keypoints_1, 
        std::vector<cv::Point2f>& keypoints_2);

    // triangulate
    void TriangulateKeypoints(
        const std::vector<cv::Point2f>& keypoints_1, 
        const std::vector<cv::Point2f>& keypoints_2, 
        std::vector<cv::Mat>& points);

    Eigen::Matrix4f CalcTransformation();

private:
    cv::Ptr<cv::FeatureDetector>        m_detector;
    cv::Ptr<cv::DescriptorMatcher>      m_matcher;

    CameraModel::Stereo*                m_camera;

    cv::Mat                             m_img_1;
    cv::Mat                             m_img_2;
    std::vector<cv::KeyPoint>           m_keypoints_1;
    std::vector<cv::KeyPoint>           m_keypoints_2;
    std::vector<cv::Mat>                m_points_1;
    std::vector<cv::Mat>                m_points_2;
    std::vector<cv::Mat>                m_descriptors_1;
    std::vector<cv::Mat>                m_descriptors_2;

    cv::Mat*                            m_prev_img;
    cv::Mat*                            m_curr_img;
    std::vector<cv::KeyPoint>*          m_curr_keypoints;
    std::vector<cv::KeyPoint>*          m_prev_keypoints;
    std::vector<cv::Mat>*               m_curr_points;
    std::vector<cv::Mat>*               m_prev_points;
    std::vector<cv::Mat>*               m_curr_descriptors;
    std::vector<cv::Mat>*               m_prev_descriptors;

    std::vector<cv::Mat>                m_poses;

    bool                                m_initialized = false;
    bool                                m_success = true;

    RANSAC::Solver<PointPair, Eigen::Matrix4f>* m_solver;
    PointSetTransModel                  m_trans_model;
    Eigen::Matrix4f                     m_trans;
};
