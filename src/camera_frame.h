#pragma once

#include "math_utils.h"
#include "map_point.h"
#include "observation.h"

#include <Eigen/Geometry>
#include "opencv4/opencv2/features2d.hpp"


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
            m_pose_glb = prev_pose * m_pose_rel;
            Normalize(m_pose_glb);
        }
        else m_pose_glb = m_pose_rel;
    }

    Eigen::Matrix4f GlobalPose() const { return m_pose_glb; }
    Eigen::Matrix4f RelativePose() const { return m_pose_rel; }

    void GlobalPose(const Eigen::Matrix4f& pose)
    {
        Eigen::Matrix4f pose_diff = pose * m_pose_glb.inverse();
        this->TransformMapPoints(pose_diff);

        m_pose_glb = pose;
        Normalize(m_pose_glb);
        
        // update relative pose
        Eigen::Matrix4f pose_prev = Eigen::Matrix4f::Identity();
        if (m_prev_frame)
        {
            pose_prev = m_prev_frame->GlobalPose();
        }

        m_pose_rel = pose_prev.inverse() * m_pose_glb;
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

        Eigen::Matrix4f pose_glb_tmp = m_pose_glb;

        m_pose_glb = pose_prev * m_pose_rel;
        Normalize(m_pose_glb);

        Eigen::Matrix4f pose_diff = pose_glb_tmp * m_pose_glb.inverse();
        this->TransformMapPoints(pose_diff);
    }

    void UpdatePose() { RelativePose(m_pose_rel); }

    void AddObservation(const Observation& obs) { m_observations.emplace_back(obs); }
    const std::vector<Observation>& Observations() const { return m_observations; }

    void Descriptors(const std::vector<cv::Mat>& descriptors) { m_descriptors = descriptors; }
    const std::vector<cv::Mat>& Descriptors() { return m_descriptors; }

    void Points(const std::vector<cv::Mat>& points) { m_points = points; }
    const std::vector<cv::Mat>& Points() { return m_points; }

    MapPoint* MapPointRef(int idx) { return m_point_refs[idx]; }
    std::vector<MapPoint*> MapPointRefs() { return m_point_refs; }

    void AddMapPoints(MapPoint* point, bool first_observed)
    {
        m_point_refs.emplace_back(point);
        m_point_masks.emplace_back(first_observed);
    }

    void UpdateMapPoint(int idx, int point_id, MapPoint* point_ref, bool first_observed)
    {
        // update tracked points descriptors
        MapPoint* mp = m_point_refs[idx];
        std::vector<cv::Mat> descriptors = mp->Descriptors();

        for (auto descriptor : descriptors)
        {
            point_ref->AddDescriptor(descriptor);
        }

        // update observation
        m_observations[idx].point_id = point_id;
        
        // assign new ref
        m_point_refs[idx] = point_ref;
        m_point_masks[idx] = first_observed;
    }

    void TransformMapPoints(const Eigen::Matrix4f& trans)
    {
        for (int i = 0; i < m_point_refs.size(); i++)
        {
            if (m_point_masks[i] == true)
            {
                m_point_refs[i]->Transform(trans);
            }
        }
    }

private:
    Frame*                          m_prev_frame = nullptr;

    Eigen::Matrix4f                 m_pose_glb;
    Eigen::Matrix4f                 m_pose_rel;

    // raw frame data
    std::vector<cv::Mat>            m_points;       // triangulated points (local)
    std::vector<cv::Mat>            m_descriptors;  // descriptors from left camera

    // map level data
    std::vector<MapPoint*>          m_point_refs;   // reference to points in map
    std::vector<bool>               m_point_masks;  // true if point first observed in this frame
    std::vector<Observation>        m_observations; // stereo observations from feature matching
};
