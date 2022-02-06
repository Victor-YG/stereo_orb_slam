#pragma once

#include "observation.h"

#include <Eigen/Geometry>


class Frame
{
public:
    Frame(const Eigen::Matrix4f& pose): m_pose_global(pose)
    {
        Eigen::Quaternionf q(m_pose_global.block<3, 3>(0, 0));
        m_pose_global.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    }

    void GlobalPose(const Eigen::Matrix4f& pose) { m_pose_global = pose; }
    Eigen::Matrix4f GlobalPose() const { return m_pose_global; }

    void AddObservation(const Observation& obs) { m_observations.emplace_back(obs); }
    const std::vector<Observation>& GetObservations() const { return m_observations; }

private:
    unsigned int                m_prev_frame_id;
    Eigen::Matrix4f             m_pose_global;
    std::vector<Observation>    m_observations;
};
