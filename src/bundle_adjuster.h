#pragma once

#include "camera_model.h"
#include "map_point.h"
#include "camera_frame.h"
#include "observation.h"

#include <vector>

#include "ceres/ceres.h"


class BundleAdjuster
{
public:
    BundleAdjuster(std::vector<Frame*>& cam_frames, std::vector<MapPoint>& ldm_points);

    void Optimize(unsigned int start_frame_id, unsigned int end_frame_id);

private:
    std::vector<Frame*>&                m_cam_frames;
    std::vector<MapPoint>&              m_ldm_points;

    ceres::Solver::Options              m_options;
};
