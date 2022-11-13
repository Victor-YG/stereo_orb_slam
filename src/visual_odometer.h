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
        std::vector<Frame*>& cam_frames, 
        std::vector<MapPoint*>& ldm_points);
    ~VisualOdometer();

    void Camera(CameraModel::Stereo* camera);
    CameraModel::Stereo* Camera() const;

    FrameDataContainer* GetCurrFrameData();

    static void MatchPointsBetweenFrames(
        Frame* src, Frame* dst, 
        std::vector<PointPair>& point_pairs, 
        std::vector<cv::DMatch>& final_matches);

    static void MatchPoints(
        const std::vector<cv::Mat>& descriptors_src, 
        const std::vector<cv::Mat>& descriptors_dst, 
        const std::vector<cv::Mat>& points_src, 
        const std::vector<cv::Mat>& points_dst, 
        std::vector<PointPair>& point_pairs, 
        std::vector<cv::DMatch>& final_matches);

    static bool CalcTransformation(
        const std::vector<PointPair>& point_pairs, 
        std::vector<float>& weights,
        Eigen::Matrix4f& transformation, 
        std::vector<bool>& mask, 
        std::vector<float>& losses);

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

    void MatchFeaturesBetweenTemporalFrames(
        std::vector<PointPair>& point_pairs, 
        std::vector<cv::DMatch>& final_matches);

    void Update(
        const Eigen::Matrix4f& trans, 
        const std::vector<cv::DMatch>& final_matches);

private:
    // features detection, matching, and tracking
    static inline cv::Ptr<cv::FeatureDetector>      m_detector;
    static inline cv::Ptr<cv::DescriptorMatcher>    m_matcher;

    // transformation estimation
    static inline RANSAC::Solver<PointPair, Eigen::Matrix4f>* m_solver;
    static inline PointSetTransModel                m_trans_model;

    // frame data ref
    FrameDataContainer*                 m_curr_container;
    FrameDataContainer*                 m_prev_container;

    // frame data
    FrameDataContainer                  m_container_1;
    FrameDataContainer                  m_container_2;

    // camera
    CameraModel::Stereo*                m_camera;
    Eigen::Matrix4f                     m_pose;
    float                               m_max_distance;

    // status
    bool                                m_initialized = false;
    bool                                m_success = true;

    // camera and landmark tracking
    std::vector<Frame*>&                m_cam_frames;
    std::vector<MapPoint*>&             m_ldm_points;
};
