#pragma once

#include <array>
#include <vector>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/calib3d.hpp"


namespace CameraModel
{

class PinholeCamera
{
public:
    PinholeCamera() {};
    PinholeCamera(float fx, float fy, float cx, float cy, float d[5], float pose[16]);

    cv::Mat GetCameraMatrix() const;
    cv::Mat GetDistortionCoef() const;
    cv::Mat GetPoseMatrix() const;
    cv::Mat GetProjectionMatrix() const;
    
    void UndistortPoints(
        const std::vector<cv::Point2f>& keypoints,
        std::vector<cv::Point2f>& undistorted_keypoints) const;

protected:
    cv::Mat     m_camera_mat;
    cv::Mat     m_distortion;
    cv::Mat     m_pose_mat;         // camera pose relative to robot base
    cv::Mat     m_projection_mat;
};


class Stereo
{
public:
    Stereo(const PinholeCamera& cam_1, const PinholeCamera& cam_2);

    PinholeCamera* GetCamera1();
    PinholeCamera* GetCamera2();

    float MaxSensibleDistance();

    void Triangulate(
        const std::vector<cv::Point2f>& keypoints_1,
        const std::vector<cv::Point2f>& keypoints_2,
        std::vector<cv::Mat>& points);

protected:
    PinholeCamera m_cam_1;
    PinholeCamera m_cam_2;
};


class StereoRectified: public Stereo
{
public:
    StereoRectified(
        const PinholeCamera& cam_l, 
        const PinholeCamera& cam_r);

    void Reprojection(
        const float& u, 
        const float& v, 
        const float& d, 
        float& x, 
        float& y, 
        float& z);

    void Triangulate(
        const std::vector<cv::Point2f>& keypoints_l,
        const std::vector<cv::Point2f>& keypoints_r,
        std::vector<cv::Mat>& points);

protected:
    float m_rect_l[12];
    float m_rect_r[12];
    float m_reprojection[16];
    float m_transformation[16];
};

} // CameraModel
