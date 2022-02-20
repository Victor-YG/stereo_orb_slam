#include "bundle_adjuster.h"

#include <string>

#include "math_utils.h"
#include "reprojection_error.h"


BundleAdjuster::BundleAdjuster(std::vector<Frame*>& cam_frames, std::vector<MapPoint>& ldm_points):
    m_cam_frames(cam_frames), m_ldm_points(ldm_points)
{
    // setup optimizer options
    m_options.max_num_iterations = 10;
    m_options.minimizer_progress_to_stdout = true;
    m_options.num_threads = 1;
    m_options.eta = 1e-2;
    m_options.max_solver_time_in_seconds = 1e32;
    m_options.use_nonmonotonic_steps = false;
    m_options.minimizer_type = ceres::TRUST_REGION;
    m_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    m_options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
    m_options.use_inner_iterations = false;

    // setup linear solver options
    m_options.linear_solver_type = ceres::SPARSE_SCHUR;
    m_options.preconditioner_type = ceres::JACOBI;
    m_options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
    m_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    m_options.dense_linear_algebra_library_type = ceres::EIGEN;
    m_options.use_explicit_schur_complement = false;
    m_options.use_mixed_precision_solves = false;
    m_options.max_num_refinement_iterations = 10;
    
    m_options.gradient_tolerance = 1e-16;
    m_options.function_tolerance = 1e-16;
}

void BundleAdjuster::Optimize(unsigned int start_frame_id, unsigned int end_frame_id)
{
    std::vector<std::array<double, 6>> camera_poses;
    camera_poses.reserve(end_frame_id - start_frame_id);

    std::vector<std::array<double, 3>> points;
    std::vector<unsigned int> point_ids;

    // rough count num of points
    unsigned int num_of_points = 0;
    for (int i = start_frame_id; i < end_frame_id; i++)
    {
        Frame* frame = m_cam_frames[i];
        num_of_points += frame->GetObservations().size(); // more obs than points
    }

    points.reserve(num_of_points);
    point_ids.reserve(num_of_points);
    std::map<unsigned int, unsigned int> glb_id_to_pt_idx;

    // create problem
    ceres::Problem problem;

    for (int frame_id = start_frame_id; frame_id < end_frame_id; frame_id++)
    {
        // add camera pose
        Frame* frame = m_cam_frames[frame_id];
        Eigen::Matrix4f mat = frame->GlobalPose().inverse();
        std::array<double, 6> pose;
        MatrixToPose(mat, pose);
        camera_poses.emplace_back(pose);

        // add points and loss from observations
        const std::vector<Observation>& observations = frame->GetObservations();

        for (int i = 0; i < observations.size(); i++)
        {
            // add point
            unsigned int pt_idx = 0;
            Observation obs = observations[i];
            if (glb_id_to_pt_idx.find(obs.point_id) == glb_id_to_pt_idx.end())
            {
                MapPoint* mp = &m_ldm_points[obs.point_id];

                std::array<float, 3> position = mp->Position();
                std::array<double, 3> position_d;
                position_d[0] = position[0];
                position_d[1] = position[1];
                position_d[2] = position[2];

                points.emplace_back(position_d);
                point_ids.emplace_back(obs.point_id);

                pt_idx = points.size() - 1;
                glb_id_to_pt_idx[obs.point_id] = pt_idx;
            }
            else pt_idx = glb_id_to_pt_idx[obs.point_id];

            // add loss
            ceres::CostFunction* cost_func;
            cost_func = ReprojectionError::Create(obs.u_l, obs.v_l, obs.u_r, obs.v_r);
            ceres::LossFunction* loss_func = new ceres::HuberLoss(1.0);

            problem.AddResidualBlock(cost_func, loss_func, &camera_poses[camera_poses.size() - 1][0], &points[pt_idx][0]);
            
            for (int k = 0; k < 3; k++)
            {
                problem.SetParameterLowerBound(&points[pt_idx][0], k, -1000);
                problem.SetParameterUpperBound(&points[pt_idx][0], k,  1000);
            }
        }
    }

    // fix first camera frame
    problem.SetParameterBlockConstant(&camera_poses[0][0]);

    // solve problem
    ceres::Solver::Summary summary;
    ceres::Solve(m_options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // update result
    for (int i = start_frame_id; i < end_frame_id; i++)
    {
        Eigen::Matrix4f trans;
        PoseToMatrix(camera_poses[i - start_frame_id], trans);
        m_cam_frames[i]->GlobalPose(trans.inverse());
    }

    for (int i = 0; i < points.size(); i++)
    {
        unsigned int global_id = point_ids[i];
        m_ldm_points[global_id].Position(points[i]);
    }
}
