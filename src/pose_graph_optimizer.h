#pragma once

#include "pose_graph.h"
#include "camera_frame.h"
#include "bundle_adjuster.h"
#include "stereo_reprojection.h"

#include <vector>

#include "g2o/core/block_solver.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/optimization_algorithm_levenberg.h"


class PoseGraphOptimizer
{
public:
    PoseGraphOptimizer(
        BundleAdjuster& ba,
        std::vector<Frame*>& cam_frames,
        std::vector<PoseGraphEdge>& edges);

    ~PoseGraphOptimizer();

    void Optimize();

private:
    void AddOdometryConstraints(unsigned int start_frame_id, unsigned int end_frame_id);
    void AddLoopClosureConstraints();
    void MatchFeaturesBetweenOverlapedFrames(int src_id, int dst_id, std::vector<ObservationPair>& point_pairs);

    void SavePoseGraph(const std::string file_path);

private:
    BundleAdjuster&                     m_ba;
    std::vector<Frame*>&                m_cam_frames;
    std::vector<PoseGraphEdge>&         m_loop_edges;

    g2o::SparseOptimizer                m_optimizer;
    g2o::RobustKernelHuber              m_kernel;
    std::vector<g2o::VertexSE3*>        m_vertices;
    std::vector<g2o::EdgeSE3*>          m_edges;

    unsigned int                        m_last_id = 0;

    Eigen::Matrix<double, 6, 6>         m_information;
};
