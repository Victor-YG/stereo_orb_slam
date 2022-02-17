#pragma once

#include "math_utils.h"
#include "observation.h"

#include <Eigen/Geometry>


class Frame
{
public:
    Frame(Frame* prev_frame, const Eigen::Matrix4f& pose_rel)
    {
        m_prev_frame = prev_frame;
        m_pose_rel = pose_rel;
        Normalize(m_pose_rel);

        if (m_prev_frame)
        {
            Eigen::Matrix4f prev_pose = m_prev_frame->GlobalPose();
            m_pose_abs = prev_pose * m_pose_rel;
            Normalize(m_pose_abs);
        }
        else
        {
            m_pose_abs = m_pose_rel;
        }
    }

    Eigen::Matrix4f GlobalPose() const { return m_pose_abs; }
    Eigen::Matrix4f RelativePose() const { return m_pose_rel; }

    void GlobalPose(const Eigen::Matrix4f& pose)
    {
        m_pose_abs = pose;
        Normalize(m_pose_abs);
        
        // update relative pose
        Eigen::Matrix4f pose_prev = Eigen::Matrix4f::Identity();
        if (m_prev_frame)
        {
            pose_prev = m_prev_frame->GlobalPose();
        }

        m_pose_rel = pose_prev.inverse() * m_pose_abs;
        Normalize(m_pose_rel);
    }

    void RelativePose(const Eigen::Matrix4f& pose)
    {
        m_pose_rel = pose;
        Normalize(m_pose_rel);

        // update global pose
        Eigen::Matrix4f pose_prev = Eigen::Matrix4f::Identity();
        if (m_prev_frame)
        {
            pose_prev = m_prev_frame->GlobalPose();
        }

        m_pose_abs = pose_prev * m_pose_rel;
        Normalize(m_pose_abs);
    }

    void AddObservation(const Observation& obs) { m_observations.emplace_back(obs); }
    const std::vector<Observation>& GetObservations() const { return m_observations; }

private:
    Frame*                      m_prev_frame = nullptr;
    unsigned int                m_prev_frame_id;
    Eigen::Matrix4f             m_pose_abs;
    Eigen::Matrix4f             m_pose_rel;
    std::vector<Observation>    m_observations;
};
