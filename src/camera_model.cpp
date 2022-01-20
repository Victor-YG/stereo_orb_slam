#include "camera_model.h"

#include <cstring>
#include <assert.h>

#include "opencv4/opencv2/features2d.hpp"


namespace CameraModel
{

// PinholeCamera
PinholeCamera::PinholeCamera(float fx, float fy, float cx, float cy, float d[5], float pose[16])
{
    m_camera_mat = cv::Mat::zeros(3, 3, CV_32F);
    m_camera_mat.at<float>(0, 0) = fx;
    m_camera_mat.at<float>(0, 2) = cx;
    m_camera_mat.at<float>(1, 1) = fy;
    m_camera_mat.at<float>(1, 2) = cy;
    m_camera_mat.at<float>(2, 2) = 1.0;

    m_distortion = cv::Mat(1, 5, CV_32F);
    m_distortion.at<float>(0) = d[0]; // k1
    m_distortion.at<float>(1) = d[1]; // k2
    m_distortion.at<float>(2) = d[2]; // p1
    m_distortion.at<float>(3) = d[3]; // p2
    m_distortion.at<float>(4) = d[4]; // k3

    m_pose_mat = cv::Mat::zeros(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
    {
        m_pose_mat.at<float>(i, 0) = pose[i * 4 + 0];
        m_pose_mat.at<float>(i, 1) = pose[i * 4 + 1];
        m_pose_mat.at<float>(i, 2) = pose[i * 4 + 2];
        m_pose_mat.at<float>(i, 3) = pose[i * 4 + 3];
    }

    cv::Mat mat = cv::Mat::zeros(3, 4, CV_32F);
    mat.at<float>(0, 0) = fx;
    mat.at<float>(0, 1) = 0.0;
    mat.at<float>(0, 2) = cx;
    mat.at<float>(1, 0) = 0.0;
    mat.at<float>(1, 1) = fy;
    mat.at<float>(1, 2) = cy;
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = 0.0;
    mat.at<float>(2, 2) = 1.0;

    m_projection_mat = mat * m_pose_mat;
}

cv::Mat PinholeCamera::GetCameraMatrix() const
{
    return m_camera_mat;
}

cv::Mat PinholeCamera::GetDistortionCoef() const
{
    return m_distortion;
}

cv::Mat PinholeCamera::GetPoseMatrix() const
{
    return m_pose_mat;
}

cv::Mat PinholeCamera::GetProjectionMatrix() const
{  
    return m_projection_mat;
}

void PinholeCamera::UndistortPoints(
    const std::vector<cv::Point2f>& keypoints,
    std::vector<cv::Point2f>& undistorted_keypoints) const
{
    cv::undistortPoints(
        keypoints, 
        undistorted_keypoints, 
        m_camera_mat, 
        m_distortion, 
        cv::noArray(), 
        cv::noArray());

    for (int i = 0; i < undistorted_keypoints.size(); i++)
    {
        cv::Point2f* kp = &undistorted_keypoints[i];
        kp->x = m_camera_mat.at<float>(0, 0) * kp->x + m_camera_mat.at<float>(0, 2); // fx * u + cx
        kp->y = m_camera_mat.at<float>(1, 1) * kp->y + m_camera_mat.at<float>(1, 2); // fy * v + cy
    }
}

// Stereo
Stereo::Stereo(
    const PinholeCamera& cam_1, 
    const PinholeCamera& cam_2)
{
    m_cam_1 = cam_1;
    m_cam_2 = cam_2;
}

void Stereo::Triangulate(
    const std::vector<cv::Point2f>& keypoints_1,
    const std::vector<cv::Point2f>& keypoints_2,
    std::vector<cv::Mat>& points)
{
    // undistort
    std::vector<cv::Point2f> undistorted_kp_1;
    std::vector<cv::Point2f> undistorted_kp_2;
    m_cam_1.UndistortPoints(keypoints_1, undistorted_kp_1);
    m_cam_2.UndistortPoints(keypoints_2, undistorted_kp_2);

    // triangulate
    cv::Mat points_homo;
    cv::Mat projection_1 = m_cam_1.GetProjectionMatrix();
    cv::Mat projection_2 = m_cam_2.GetProjectionMatrix();
    cv::triangulatePoints(projection_1, projection_2, undistorted_kp_1, undistorted_kp_2, points_homo);

    int N = points_homo.cols;
    points.reserve(N);
    
    for (int i = 0; i < N; i++)
    {
        cv::Mat point = cv::Mat(1, 3, CV_32F);
        point.at<float>(0, 0) = points_homo.at<float>(0, i) / points_homo.at<float>(3, i);
        point.at<float>(0, 1) = points_homo.at<float>(1, i) / points_homo.at<float>(3, i);
        point.at<float>(0, 2) = points_homo.at<float>(2, i) / points_homo.at<float>(3, i);
        points.emplace_back(point);
    }
}

// StereoRectified
StereoRectified::StereoRectified(
        const PinholeCamera& cam_l, 
        const PinholeCamera& cam_r)
        : Stereo(cam_l, cam_r)
{
    cv::Mat cam_mat_l = cam_l.GetCameraMatrix();
    cv::Mat cam_mat_r = cam_r.GetCameraMatrix();
    float fx_l = cam_mat_l.at<float>(0, 0);
    float fy_l = cam_mat_l.at<float>(1, 1);
    float cx_l = cam_mat_l.at<float>(0, 2);
    float cy_l = cam_mat_l.at<float>(1, 2);
    float fx_r = cam_mat_r.at<float>(0, 0);
    float fy_r = cam_mat_r.at<float>(1, 1);
    float cx_r = cam_mat_r.at<float>(0, 2);
    float cy_r = cam_mat_r.at<float>(1, 2);
    assert(fx_l == fy_l == fx_r == fy_r && cy_l == cy_r);

    cv::Mat pose_mat_l = cam_l.GetPoseMatrix();
    cv::Mat pose_mat_r = cam_r.GetPoseMatrix();
    cv::Mat trans = pose_mat_l.inv() * pose_mat_r;
    float baseline = trans.at<float>(1, 3);

    m_reprojection[ 0] = 1.0;
    m_reprojection[ 1] = 0.0;
    m_reprojection[ 2] = 0.0;
    m_reprojection[ 3] = - cx_l;
    m_reprojection[ 4] = 0.0;
    m_reprojection[ 5] = 1.0;
    m_reprojection[ 6] = 0.0;
    m_reprojection[ 7] = - cy_l;
    m_reprojection[ 8] = 0.0;
    m_reprojection[ 9] = 0.0;
    m_reprojection[10] = 0.0;
    m_reprojection[11] = fx_l;
    m_reprojection[12] = 0.0;
    m_reprojection[13] = 0.0;
    m_reprojection[14] = - 1.0 / baseline;
    m_reprojection[15] = (cx_l - cx_r) / baseline;
}

void StereoRectified::Reprojection(const float& u, const float& v, const float& d, float& x, float& y, float& z)
{
    float fw_inv = 1.0 / (d * m_reprojection[14]) + m_reprojection[15];
    x = fw_inv * (u * m_reprojection[0] + m_reprojection[3]);
    y = fw_inv * (v * m_reprojection[5] + m_reprojection[7]);
    z = m_reprojection[11] * fw_inv;
}

void StereoRectified::Triangulate(
    const std::vector<cv::Point2f>& keypoints_l,
    const std::vector<cv::Point2f>& keypoints_r,
    std::vector<cv::Mat>& points)
{
    assert(keypoints_l.size() == keypoints_r.size());

    for (int i = 0; i < keypoints_l.size(); i++)
    {
        float u = keypoints_l[i].x;
        float v = keypoints_l[i].y;
        float d = keypoints_l[i].x - keypoints_r[i].x;
        
        float x, y, z;
        this->Reprojection(u, v, d, x, y, z);

        cv::Mat point = cv::Mat(1, 3, CV_32F);
        point.at<float>(0) = x;
        point.at<float>(1) = y;
        point.at<float>(2) = z;

        points.emplace_back(point);
    }
}

} // CameraModel