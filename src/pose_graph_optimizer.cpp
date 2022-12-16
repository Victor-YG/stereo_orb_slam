#include "pose_graph_optimizer.h"

#include "visual_odometer.h"


PoseGraphOptimizer::PoseGraphOptimizer(
    BundleAdjuster& ba,
    std::vector<Frame*>& cam_frames,
    std::vector<PoseGraphEdge>& edges):
    m_ba(ba),
    m_cam_frames(cam_frames),
    m_loop_edges(edges)
{
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );

    m_optimizer.setAlgorithm(solver);
    m_optimizer.setVerbose(true);

    m_information = Eigen::Matrix<double, 6, 6>::Identity();
    m_information(0, 0) = 0.01;
    m_information(1, 1) = 0.01;
    m_information(2, 2) = 0.01;
}

PoseGraphOptimizer::~PoseGraphOptimizer()
{
    // SavePoseGraph("./pose_graph.txt");
}

void PoseGraphOptimizer::Optimize()
{
    // determine range of camera frame
    unsigned int start_id = m_cam_frames.size();
    unsigned int   end_id = 0;

    if (m_loop_edges.size() == 0)
    { // global pose graph optimization
        start_id = 0;
        end_id = m_cam_frames.size() - 1;
    }

    else
    { // partial pose graph optimization
        for (auto edge : m_loop_edges)
        {
            start_id = (edge.first > start_id) ? start_id : edge.first;
            end_id = (edge.second < end_id) ? end_id : edge.second;
        }
    }

    // add constraints
    AddOdometryConstraints(m_last_id, end_id);
    AddLoopClosureConstraints();

    m_last_id = end_id;

    bool has_gauge_freedom = m_optimizer.gaugeFreedom();
    if (has_gauge_freedom) {
        g2o::OptimizableGraph::Vertex* gauge_node = m_optimizer.findGauge();
        gauge_node->setFixed(true);
    }

    // optimize
    m_optimizer.initializeOptimization();
    m_optimizer.optimize(10);

    // update
    for (int i = 0; i < m_vertices.size(); i++)
    {
        std::array<double, 7> z;
        m_vertices[i]->getEstimateData(&z[0]);

        Eigen::Quaternionf q(z[6], z[3], z[4], z[5]);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose(0, 3) = z[0];
        pose(1, 3) = z[1];
        pose(2, 3) = z[2];

        m_cam_frames[i]->GlobalPose(pose);
    }

    // propagate
    for (int i = m_vertices.size(); i < m_cam_frames.size(); i++)
    {
        m_cam_frames[i]->UpdatePose();
    }

    // global ba
    m_ba.Optimize(0, m_cam_frames.size() - 1);
}

void PoseGraphOptimizer::AddOdometryConstraints(unsigned int start_frame_id, unsigned int end_frame_id)
{
    std::vector<std::array<double, 7>> glb_poses;

    // add first vertex
    Frame* frame = m_cam_frames[start_frame_id];
    Eigen::Matrix4f pose_glb = frame->GlobalPose();

    Eigen::Quaternionf q(pose_glb.block<3, 3>(0, 0));
    Eigen::Vector3f    t(pose_glb.block<3, 1>(0, 3));
    std::array<double, 7> z;
    z[0] = t(0), z[1] = t(1), z[2] = t(2);
    z[3] = q.x(), z[4] = q.y(), z[5] = q.z(), z[6] = q.w();
    glb_poses.emplace_back(z);

    g2o::VertexSE3* v_SE3 = new g2o::VertexSE3();

    v_SE3->setId(start_frame_id);
    v_SE3->setEstimateDataImpl(&z[0]);

    if (start_frame_id == 0)
    {
        v_SE3->setFixed(true);
    }

    m_optimizer.addVertex(v_SE3);
    m_vertices.emplace_back(v_SE3);

    for (int i = start_frame_id + 1; i <= end_frame_id; i++)
    {
        Frame* frame = m_cam_frames[i];

        // add vertex
        Eigen::Matrix4f pose_glb = frame->GlobalPose();
        Eigen::Matrix4f pose_rel = frame->RelativePose();

        q = Eigen::Quaternionf(pose_glb.block<3, 3>(0, 0));
        t = Eigen::Vector3f(pose_glb.block<3, 1>(0, 3));
        z[0] = t(0), z[1] = t(1), z[2] = t(2);
        z[3] = q.x(), z[4] = q.y(), z[5] = q.z(), z[6] = q.w();
        glb_poses.emplace_back(z);

        g2o::VertexSE3* v_SE3 = new g2o::VertexSE3();

        v_SE3->setId(i);
        v_SE3->setEstimateDataImpl(&z[0]);

        m_optimizer.addVertex(v_SE3);
        m_vertices.emplace_back(v_SE3);

        // add edge
        g2o::EdgeSE3* e_SE3 = new g2o::EdgeSE3();
        e_SE3->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            m_optimizer.vertices().find(i - 1)->second)
        );
        e_SE3->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            m_optimizer.vertices().find(i)->second)
        );

        q = Eigen::Quaternionf(pose_rel.block<3, 3>(0, 0));
        t = Eigen::Vector3f(pose_rel.block<3, 1>(0, 3));
        z[0] = t(0), z[1] = t(1), z[2] = t(2);
        z[3] = q.x(), z[4] = q.y(), z[5] = q.z(), z[6] = q.w();

        e_SE3->setMeasurementData(&z[0]);
        double* info = e_SE3->informationData();
        Eigen::Matrix<double, 6, 6> info_mat(info);
        e_SE3->setInformation(m_information);

        e_SE3->setRobustKernel(&m_kernel);
        e_SE3->setParameterId(0, 0);

        m_optimizer.addEdge(e_SE3);
        m_edges.emplace_back(e_SE3);
    }
}

void PoseGraphOptimizer::AddLoopClosureConstraints()
{
    for (auto edge : m_loop_edges)
    {
        unsigned int id_1 = edge.first;
        unsigned int id_2 = edge.second;
        std::vector<PointPair> point_pairs;
        MatchFeaturesBetweenOverlapedFrames(id_1, id_2, point_pairs);
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();

        unsigned int N = point_pairs.size();
        std::vector<bool> mask(N, false);
        std::vector<float> weights(N, 1.0);
        std::vector<float> losses(N, 0.0);
        // bool success = VisualOdometer::CalcTransformation(point_pairs, weights, trans, mask, losses);
        bool success = false;
        if (!success) continue;

        g2o::EdgeSE3* e_SE3 = new g2o::EdgeSE3();
        e_SE3->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            m_optimizer.vertices().find(id_1)->second)
        );
        e_SE3->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            m_optimizer.vertices().find(id_2)->second)
        );

        Eigen::Quaternionf q(trans.block<3, 3>(0, 0));
        Eigen::Vector3f t(trans.block<3, 1>(0, 3));
        std::array<double, 7> z;
        z[0] = t(0), z[1] = t(1), z[2] = t(2);
        z[3] = q.x(), z[4] = q.y(), z[5] = q.z(); z[6] = q.w();

        e_SE3->setMeasurementData(&z[0]);
        e_SE3->setInformation(m_information);

        e_SE3->setRobustKernel(&m_kernel);
        e_SE3->setParameterId(0, 0);

        m_optimizer.addEdge(e_SE3);
        m_edges.emplace_back(e_SE3);
    }

    m_loop_edges.clear();
}

void PoseGraphOptimizer::MatchFeaturesBetweenOverlapedFrames(int src_id, int dst_id, std::vector<PointPair>& point_pairs)
{
    Frame* frame_src = m_cam_frames[src_id];
    Frame* frame_dst = m_cam_frames[dst_id];
    std::vector<cv::Mat> des_src = frame_src->Descriptors();
    std::vector<cv::Mat> des_dst = frame_dst->Descriptors();
    std::vector<cv::Mat> points_src = frame_src->Points();
    std::vector<cv::Mat> points_dst = frame_dst->Points();

    std::vector<cv::DMatch> final_matches;
    VisualOdometer::MatchPoints(des_src, des_dst, points_src, points_dst, point_pairs, final_matches);

    for (auto match : final_matches)
    {
        unsigned int idx_src = match.trainIdx;
        unsigned int idx_dst = match.queryIdx;

        unsigned int point_id = frame_src->Observations()[idx_src].point_id;
        MapPoint* mp = frame_src->MapPointRef(idx_src);
        frame_dst->UpdateMapPoint(idx_dst, point_id, mp, false);
    }

    std::cout << "[INFO]: Matched " << point_pairs.size() << " points for loop closure." << std::endl;
}

void PoseGraphOptimizer::SavePoseGraph(const std::string file_path)
{
    std::ofstream output;
    output.open(file_path);

    if (!output.is_open()) return;

    // write header
    output << m_vertices.size() << " " << m_edges.size() << std::endl;

    // write vertices
    for (int i = 0; i < m_vertices.size(); i++)
    {
        g2o::VertexSE3* vertex = m_vertices[i];
        std::array<double, 7> data;
        vertex->getEstimateData(&data[0]);

        output << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << " "
               << data[4] << " " << data[5] << " " << data[6] << "\n";
    }

    // write edges
    for (int i = 0; i < m_edges.size(); i++)
    {
        g2o::EdgeSE3* edge = m_edges[i];
        std::array<double, 7> data;
        edge->getMeasurementData(&data[0]);

        int src_id = edge->vertex(0)->id();
        int dst_id = edge->vertex(1)->id();

        output <<  src_id << " " <<  dst_id << " "
               << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << " "
               << data[4] << " " << data[5] << " " << data[6] << "\n";
    }
}