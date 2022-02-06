#pragma once

#include "camera_model.h"
#include "map_point.h"
#include "camera_frame.h"
#include "observation.h"
#include "absolute_orientation.h"
#include "frame_data_container.h"

#include <Eigen/Geometry>
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/features2d.hpp"


class VisualOdometer
{
public:
    VisualOdometer(
        std::vector<Frame>& cam_frames, 
        std::vector<MapPoint>& ldm_points);
    ~VisualOdometer();

    void Camera(CameraModel::Stereo* camera);
    CameraModel::Stereo* Camera();

    Eigen::Matrix4f Track(const cv::Mat& img_l, const cv::Mat& img_r);

private:
    // detect and match features across stereo images
    void MatchFeaturesBetweenStereoImages(
        const cv::Mat& img_1, 
        const cv::Mat& img_2, 
        std::vector<cv::KeyPoint>& keypoints_1, 
        std::vector<cv::KeyPoint>& keypoints_2, 
        std::vector<cv::Mat>& descriptor_1, 
        std::vector<cv::Mat>& descriptor_2);

    // triangulate
    void TriangulateKeypoints(
        const std::vector<cv::KeyPoint>& keypoints_1, 
        const std::vector<cv::KeyPoint>& keypoints_2, 
        std::vector<cv::Mat>& points);

    void MatchFeaturesBetweenTemporaryFrames(
        std::vector<PointPair>& point_pairs, 
        std::vector<cv::DMatch>& final_matches);

    Eigen::Matrix4f CalcTransformation(const std::vector<PointPair>& point_pairs);

    void Update(
        const Eigen::Matrix4f& trans, 
        const std::vector<cv::DMatch>& final_matches);

private:
    // camera
    CameraModel::Stereo*                m_camera;
    Eigen::Matrix4f                     m_pose;

    // features detection, matching, and tracking
    cv::Ptr<cv::FeatureDetector>        m_detector;
    cv::Ptr<cv::DescriptorMatcher>      m_matcher;

    FrameDataContainer                  m_container_1;
    FrameDataContainer                  m_container_2;

    FrameDataContainer*                  m_curr_container;
    FrameDataContainer*                  m_prev_container;

    bool                                m_initialized = false;
    bool                                m_success = true;

    // camera and landmark tracking
    std::vector<Frame>&                 m_cam_frames;
    std::vector<MapPoint>&              m_ldm_points;

    // transformation estimation
    RANSAC::Solver<PointPair, Eigen::Matrix4f>* m_solver;
    PointSetTransModel                  m_trans_model;
};
