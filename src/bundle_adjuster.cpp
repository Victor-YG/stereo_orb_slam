#include "bundle_adjuster.h"


BundleAdjuster::BundleAdjuster(std::vector<Frame>& cam_frames, std::vector<MapPoint>& ldm_points):
    m_cam_frames(cam_frames), m_ldm_points(ldm_points) { }

void BundleAdjuster::Callback_Optimize(unsigned int start_frame_idx, unsigned int end_frame_idx)
{

}


void BundleAdjuster::CreateProblem()
{

}

void BundleAdjuster::SolveProblem()
{
    
}