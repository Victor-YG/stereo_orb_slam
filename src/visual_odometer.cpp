#include "visual_odometer.h"

#include "math_utils.h"


VisualOdometer::VisualOdometer(
    std::vector<Frame*>& cam_frames,
    std::vector<MapPoint*>& ldm_points):
    m_cam_frames(cam_frames),
    m_ldm_points(ldm_points)
{
    m_detector = cv::ORB::create(1000);
    m_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    m_prev_container = &m_container_1;
    m_curr_container = &m_container_2;

    m_solver = new RANSAC::Solver<PointPair, Eigen::Matrix4f>(&m_trans_model, 100);
    m_pose = Eigen::Matrix4f::Identity(4, 4);

    cv::namedWindow("Stereo", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Temporal", cv::WINDOW_AUTOSIZE);
}

VisualOdometer::~VisualOdometer()
{
    delete m_solver;

    cv::destroyWindow("Stereo");
    cv::destroyWindow("Temporal");
}

void VisualOdometer::Camera(CameraModel::Stereo* camera)
{
    m_camera = camera;
    m_max_distance = m_camera->MaxSensibleDistance();
}

CameraModel::Stereo* VisualOdometer::Camera() const
{
    return m_camera;
}

FrameDataContainer* VisualOdometer::GetCurrFrameData()
{
    return m_prev_container;
}

void VisualOdometer::MatchPointsBetweenFrames(
    Frame* src, Frame* dst,
    std::vector<PointPair>& point_pairs,
    std::vector<cv::DMatch>& final_matches)
{
    std::vector<cv::Mat> des_dst = src->Descriptors();
    std::vector<cv::Mat> des_src = dst->Descriptors();
    std::vector<cv::Mat> points_src = src->Points();
    std::vector<cv::Mat> points_dst = dst->Points();

    MatchPoints(des_src, points_src, des_dst, points_dst, point_pairs, final_matches);
    std::cout << "[INFO]: Matched " << point_pairs.size() << " between two frames." << std::endl;
}

void VisualOdometer::MatchPoints(
    const std::vector<cv::Mat>& descriptors_src,
    const std::vector<cv::Mat>& descriptors_dst,
    const std::vector<cv::Mat>& points_src,
    const std::vector<cv::Mat>& points_dst,
    std::vector<PointPair>& point_pairs,
    std::vector<cv::DMatch>& final_matches)
{
    cv::Mat des_src, des_dst;
    for (auto descriptor : descriptors_src) des_src.push_back(descriptor);
    for (auto descriptor : descriptors_dst) des_dst.push_back(descriptor);

    // match points
    std::vector<std::vector<cv::DMatch>> matches;
    m_matcher->knnMatch(des_dst, des_src, matches, 2);

    // conduct Lowe's ratio test
    const float dist_thres = 30.0;
    const float ratio_thres = 0.5;
    for (int i = 0; i < matches.size(); i++)
    {
        float d1 = matches[i][0].distance;
        float d2 = matches[i][1].distance;

        if (d1 < dist_thres && d1 / d2 < ratio_thres)
        {
            unsigned int idx_dst = matches[i][0].queryIdx;
            unsigned int idx_src = matches[i][0].trainIdx;

            cv::Mat pt_dst = points_dst[idx_dst];
            cv::Mat pt_src = points_src[idx_src];

            Eigen::Vector3f point_dst = Eigen::Vector3f(
                pt_dst.at<float>(0),
                pt_dst.at<float>(1),
                pt_dst.at<float>(2));

            Eigen::Vector3f point_src = Eigen::Vector3f(
                pt_src.at<float>(0),
                pt_src.at<float>(1),
                pt_src.at<float>(2));

            point_pairs.emplace_back(std::make_pair(point_dst, point_src));
            final_matches.emplace_back(matches[i][0]);
        }
    }
}

bool VisualOdometer::CalcTransformation(
    const std::vector<PointPair>& point_pairs,
    std::vector<float>& weights,
    Eigen::Matrix4f& transformation,
    std::vector<bool>& mask,
    std::vector<float>& losses)
{
    if (point_pairs.size() < 10) return false;
    transformation = m_solver->Solve(point_pairs, weights, mask, losses);
    return true;
}

Eigen::Matrix4f VisualOdometer::Track(const cv::Mat& img_l, const cv::Mat& img_r)
{
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(4, 4);
    m_curr_container->image = img_l.clone();

    // stereo matching
    std::vector<cv::KeyPoint> keypoints_l;
    std::vector<cv::KeyPoint> keypoints_r;
    std::vector<cv::Mat> descriptors_l;
    std::vector<cv::Mat> descriptors_r;
    MatchFeaturesBetweenStereoImages(img_l, img_r, keypoints_l, keypoints_r, descriptors_l, descriptors_r);

    if (keypoints_l.size() < 5)
    {
        m_success = false;
        return trans;
    }

    // triangulate
    std::vector<cv::Mat> points;
    TriangulateKeypoints(keypoints_l, keypoints_r, points);

    // check points validity
    for (int i = 0; i < points.size(); i++)
    {
        // TODO::change the logic here:
        // 1. keep both the point and the descriptor
        // 2. but indicates this point is too far
        // 3. or reject far point when creating the point_pairs before CalcTransformation

        // within max range
        cv::Mat point = points[i];
        if (abs(point.at<float>(2)) < m_max_distance)
        {
            m_curr_container->AddKeyPoints(keypoints_l[i], keypoints_r[i]);
            m_curr_container->AddDescriptor(descriptors_l[i]);
            m_curr_container->AddPoint(point);
        }
        else
        {
            std::cout << "[DBUG]: Point out of bound!" << std::endl;
            std::cout << point << std::endl;
        }
    }

    std::vector<PointPair> point_pairs;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> final_matches;

    if (m_initialized)
    {
        // temporary matching
        MatchFeaturesBetweenTemporalFrames(point_pairs, matches);

        // pose estimation
        unsigned int N = point_pairs.size();
        std::vector<bool> mask(N, false);
        std::vector<float> weights(N, 1.0);
        std::vector<float> losses(N, 0.0);
        m_success = CalcTransformation(point_pairs, weights, trans, mask, losses);

        // filter matches
        for (int i = 0; i < matches.size(); i++)
        {
            if (mask[i] == true)
            {
                final_matches.emplace_back(matches[i]);
            }
        }
    }

    // update pose; add frames, landmarks, and observations
    Update(trans, final_matches);

    // update stored information
    if (m_success)
    {
        FrameDataContainer* tmp_container;
        tmp_container = m_prev_container;
        m_prev_container->Flush();
        m_prev_container = m_curr_container;
        m_curr_container = tmp_container;
    }

    m_initialized = true;
    return trans;
}

void VisualOdometer::MatchFeaturesBetweenStereoImages(
    const cv::Mat& img_1,
    const cv::Mat& img_2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::Mat>& descriptors_1,
    std::vector<cv::Mat>& descriptors_2)
{
    // extract features
    std::vector<cv::Point2f> corners_1;
    std::vector<cv::Point2f> corners_2;
    cv::goodFeaturesToTrack(img_1, corners_1, 1000, 0.01, 10, cv::noArray(), 5);
    cv::goodFeaturesToTrack(img_2, corners_2, 1000, 0.01, 10, cv::noArray(), 5);

    // refine features
    cv::Size win_size = cv::Size(5, 5);
    cv::Size zero_zone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    cv::cornerSubPix(img_1, corners_1, win_size, zero_zone, criteria);
    cv::cornerSubPix(img_2, corners_2, win_size, zero_zone, criteria);

    std::vector<cv::KeyPoint> keypoints_1_local;
    std::vector<cv::KeyPoint> keypoints_2_local;
    for (int i = 0; i < corners_1.size(); i++)
    {
        keypoints_1_local.emplace_back(cv::KeyPoint(corners_1[i], 5));
    }
    for (int i = 0; i < corners_2.size(); i++)
    {
        keypoints_2_local.emplace_back(cv::KeyPoint(corners_2[i], 5));
    }

    // compute descriptor
    cv::Mat descriptors_1_local;
    cv::Mat descriptors_2_local;
    m_detector->compute(img_1, keypoints_1_local, descriptors_1_local);
    m_detector->compute(img_2, keypoints_2_local, descriptors_2_local);

    // matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    m_matcher->knnMatch(descriptors_1_local, descriptors_2_local, knn_matches, 2);

    // conduct Lowe's ratio test
    const float dist_thres = 30.0;
    const float ratio_thres = 0.5;
    std::vector<cv::DMatch> final_matches;

    for (int i = 0; i < knn_matches.size(); i++)
    {
        auto match = knn_matches[i];

        float d1 = match[0].distance;
        float d2 = match[1].distance;

        if (d1 < dist_thres && d1 / d2 < ratio_thres)
        {
            unsigned int idx_1 = match[0].queryIdx;
            unsigned int idx_2 = match[0].trainIdx;

            keypoints_1.emplace_back(keypoints_1_local[idx_1]);
            keypoints_2.emplace_back(keypoints_2_local[idx_2]);

            descriptors_1.emplace_back(descriptors_1_local.row(idx_1));
            descriptors_2.emplace_back(descriptors_2_local.row(idx_2));

            final_matches.emplace_back(match[0]);
        }
    }

    std::cout << "[INFO]: Matched " << final_matches.size() << " points for triangulation." << std::endl;

    // display
    cv::Mat img_features;
    cv::drawMatches(img_1, keypoints_1_local, img_2, keypoints_2_local, final_matches, img_features);
    cv::imshow("Stereo", img_features);
}

void VisualOdometer::TriangulateKeypoints(
    const std::vector<cv::KeyPoint>& keypoints_1,
    const std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::Mat>& points)
{
    std::vector<cv::Point2f> keypoints_1_local;
    std::vector<cv::Point2f> keypoints_2_local;

    for (int i = 0; i < keypoints_1.size(); i++)
    {
        keypoints_1_local.emplace_back(keypoints_1[i].pt);
        keypoints_2_local.emplace_back(keypoints_2[i].pt);
    }

    m_camera->Triangulate(keypoints_1_local, keypoints_2_local, points);
}

void VisualOdometer::MatchFeaturesBetweenTemporalFrames(
    std::vector<PointPair>& point_pairs,
    std::vector<cv::DMatch>& final_matches)
{
    if (!m_initialized) return;

    // match points between curr and prev frames
    std::vector<cv::Mat> des_curr = m_curr_container->descriptors;
    std::vector<cv::Mat> des_prev = m_prev_container->descriptors;
    std::vector<cv::Mat> points_curr = m_curr_container->points;
    std::vector<cv::Mat> points_prev = m_prev_container->points;

    MatchPoints(des_prev, des_curr, points_prev, points_curr, point_pairs, final_matches);
    std::cout << "[INFO]: Matched " << final_matches.size() << " points for tracking." << std::endl;

    cv::Mat img_features;
    cv::drawMatches(
        m_curr_container->image, m_curr_container->keypoints_l,
        m_prev_container->image, m_prev_container->keypoints_l,
        final_matches, img_features);
    cv::imshow("Temporal", img_features);
}

void VisualOdometer::Update(
    const Eigen::Matrix4f& trans,
    const std::vector<cv::DMatch>& final_matches)
{
    // track points from first frame
    if (!m_initialized)
    {
        // add new frame
        Frame* new_frame = new Frame(nullptr, Eigen::Matrix4f::Identity());

        for (int i = 0; i < m_curr_container->points.size(); i++)
        {
            // add points
            cv::Mat pt = m_curr_container->points[i];
            MapPoint* mp = new MapPoint(pt.at<float>(0), pt.at<float>(1), pt.at<float>(2));
            mp->AddDescriptor(m_curr_container->descriptors[i]);

            unsigned int point_id = m_ldm_points.size();
            m_ldm_points.emplace_back(mp);
            new_frame->AddMapPoints(mp, true);
            m_curr_container->AddGlobalIndex(point_id);

            // add observations
            cv::KeyPoint kp_l = m_curr_container->keypoints_l[i];
            cv::KeyPoint kp_r = m_curr_container->keypoints_r[i];
            Observation obs = Observation(point_id, kp_l.pt.x, kp_l.pt.y, kp_r.pt.x, kp_r.pt.y, 1.0);
            new_frame->AddObservation(obs);
        }

        new_frame->Descriptors(m_curr_container->descriptors);
        new_frame->Points(m_curr_container->points);
        m_cam_frames.emplace_back(new_frame);
        return;
    }

    // add new frame
    Frame* prev_frame = m_cam_frames[m_cam_frames.size() - 1];
    Frame* new_frame = new Frame(prev_frame, trans);

    // track points from other frames
    unsigned int idx = 0;
    for (int i = 0; i < final_matches.size(); i++)
    {
        unsigned int idx_curr = final_matches[i].queryIdx;
        unsigned int idx_prev = final_matches[i].trainIdx;

        // track points first observed
        for (/*blank*/; idx < idx_curr; idx++)
        {
            // add points
            cv::Mat pt = m_curr_container->points[idx];
            MapPoint* mp = new MapPoint(pt.at<float>(0), pt.at<float>(1), pt.at<float>(2));

            mp->Transform(new_frame->GlobalPose()); // to global ref. frame
            mp->AddDescriptor(m_curr_container->descriptors[idx]);
            new_frame->AddMapPoints(mp, true);

            unsigned int point_id = m_ldm_points.size();
            m_ldm_points.emplace_back(mp);
            m_curr_container->AddGlobalIndex(point_id);

            // add observations
            cv::KeyPoint kp_l = m_curr_container->keypoints_l[idx];
            cv::KeyPoint kp_r = m_curr_container->keypoints_r[idx];
            Observation obs = Observation(point_id, kp_l.pt.x, kp_l.pt.y, kp_r.pt.x, kp_r.pt.y, 1.0);
            new_frame->AddObservation(obs);
        }

        // track point already observed
        int global_idx = m_prev_container->point_global_idx[idx_prev];
        MapPoint* mp = m_ldm_points[global_idx];

        mp->AddDescriptor(m_curr_container->descriptors[idx]);
        m_curr_container->AddGlobalIndex(global_idx);
        new_frame->AddMapPoints(mp, false);

        // add observations
        cv::KeyPoint kp_l = m_curr_container->keypoints_l[idx_curr];
        cv::KeyPoint kp_r = m_curr_container->keypoints_r[idx_curr];
        Observation obs = Observation(global_idx, kp_l.pt.x, kp_l.pt.y, kp_r.pt.x, kp_r.pt.y, 1.0);
        new_frame->AddObservation(obs);

        idx = idx_curr + 1;
    }

    // track remaining points (first observed)
    for (int i = idx; i < m_curr_container->points.size(); i++)
    {
        cv::Mat pt = m_curr_container->points[i];
        MapPoint* mp = new MapPoint(pt.at<float>(0), pt.at<float>(1), pt.at<float>(2));

        mp->Transform(new_frame->GlobalPose()); // to global ref. frame
        mp->AddDescriptor(m_curr_container->descriptors[i]);
        new_frame->AddMapPoints(mp, true);

        unsigned int point_id = m_ldm_points.size();
        m_ldm_points.emplace_back(mp);
        m_curr_container->AddGlobalIndex(point_id);

        // add observations
        cv::KeyPoint kp_l = m_curr_container->keypoints_l[i];
        cv::KeyPoint kp_r = m_curr_container->keypoints_r[i];
        Observation obs = Observation(point_id, kp_l.pt.x, kp_l.pt.y, kp_r.pt.x, kp_r.pt.y, 1.0);
        new_frame->AddObservation(obs);
    }

    new_frame->Descriptors(m_curr_container->descriptors);
    new_frame->Points(m_curr_container->points);
    m_cam_frames.emplace_back(new_frame);
}
