#pragma once

#include <Eigen/Geometry>


inline void Normalize(Eigen::Matrix4f& pose)
{
    Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
    pose.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
}

template <typename T>
inline void MatrixToPose(const Eigen::Matrix4f& mat, std::array<T, 6>& pose)
{
    Eigen::AngleAxisf rot(mat.block<3, 3>(0, 0));
    float angle = rot.angle();
    Eigen::Vector3f axis = rot.axis();

    pose[0] = axis(0) * angle;
    pose[1] = axis(1) * angle;
    pose[2] = axis(2) * angle;
    pose[3] = mat(0, 3);
    pose[4] = mat(1, 3);
    pose[5] = mat(2, 3);
}

template <typename T>
inline void PoseToMatrix(const std::array<T, 6>& pose, Eigen::Matrix4f& mat)
{
    mat = Eigen::Matrix4f::Identity();
    
    Eigen::Vector3f r(pose[0], pose[1], pose[2]);
    float f = r.norm();
    Eigen::Vector3f rn = r.normalized();
    Eigen::AngleAxisf rot(f, rn);
    mat.block<3, 3>(0, 0) = rot.toRotationMatrix();

    mat(0, 3) = pose[3];
    mat(1, 3) = pose[4];
    mat(2, 3) = pose[5];
}