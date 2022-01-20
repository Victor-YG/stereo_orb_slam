#include "visual_odometer.h"


VisualOdometer::VisualOdometer()
{
    m_detector = cv::ORB::create(1000);
    m_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    m_prev_img = &m_img_1;
    m_curr_img = &m_img_2;
    m_prev_keypoints = &m_keypoints_1;
    m_curr_keypoints = &m_keypoints_2;
    m_prev_points = &m_points_1;
    m_curr_points = &m_points_2;
    m_prev_descriptors = &m_descriptors_1;
    m_curr_descriptors = &m_descriptors_2;

    m_solver = new RANSAC::Solver<PointPair, Eigen::Matrix4f>(&m_trans_model, 20);
}

VisualOdometer::~VisualOdometer()
{
    delete m_solver;
}

void VisualOdometer::Camera(CameraModel::Stereo* camera)
{
    m_camera = camera;
}

CameraModel::Stereo* VisualOdometer::Camera()
{
    return m_camera;
}

Eigen::Matrix4f VisualOdometer::Track(const cv::Mat& img_l, const cv::Mat& img_r)
{
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(4, 4);
    *m_curr_img = img_l.clone();

    // feature matching
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> keypoints_1;
    std::vector<cv::Point2f> keypoints_2;
    MatchFeaturesBetweenStereoImages(img_l, img_r, keypoints_1, keypoints_2);

    if (keypoints_1.size() > 0)   
    {
        // triangulate
        TriangulateKeypoints(keypoints_1, keypoints_2, *m_curr_points);

        // pose estimation
        if (m_initialized) trans = CalcTransformation();
    }
    else
    {
        m_success = false;
    }

    // update stored information
    if (m_success)
    {
        cv::Mat* tmp_img;
        std::vector<cv::KeyPoint>* tmp_keypoints;
        std::vector<cv::Mat>* tmp_descriptors;
        std::vector<cv::Mat>* tmp_points;

        tmp_img = m_prev_img;
        tmp_keypoints = m_prev_keypoints;
        tmp_descriptors = m_prev_descriptors;
        tmp_points = m_prev_points;

        m_prev_keypoints->clear();
        m_prev_descriptors->clear();
        m_prev_points->clear();

        m_prev_img = m_curr_img;
        m_prev_keypoints = m_curr_keypoints;
        m_prev_descriptors = m_curr_descriptors;
        m_prev_points = m_curr_points;

        m_curr_img = tmp_img;
        m_curr_keypoints = tmp_keypoints;
        m_curr_descriptors = tmp_descriptors;
        m_curr_points = tmp_points;
    }

    m_initialized = true;
    return trans;
}

void VisualOdometer::MatchFeaturesBetweenStereoImages(
    const cv::Mat& img_1, 
    const cv::Mat& img_2, 
    std::vector<cv::Point2f>& keypoints_1, 
    std::vector<cv::Point2f>& keypoints_2)
{
    // extract features
    std::vector<cv::Point2f> corners_1;
    std::vector<cv::Point2f> corners_2;
    cv::goodFeaturesToTrack(img_1, corners_1, 1000, 0.01, 10, cv::noArray(), 5);
    cv::goodFeaturesToTrack(img_2, corners_2, 1000, 0.01, 10, cv::noArray(), 5);

    std::vector<cv::KeyPoint> kp_1;
    std::vector<cv::KeyPoint> kp_2;
    for (int i = 0; i < corners_1.size(); i++)
    {
        kp_1.emplace_back(cv::KeyPoint(corners_1[i], 5));
    }
    for (int i = 0; i < corners_2.size(); i++)
    {
        kp_2.emplace_back(cv::KeyPoint(corners_2[i], 5));
    }

    // compute descriptor
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    m_detector->compute(img_1, kp_1, descriptors_1);
    m_detector->compute(img_2, kp_2, descriptors_2);

    // matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    m_matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

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

            m_curr_keypoints->emplace_back(kp_1[idx_1]);

            cv::Mat des = descriptors_1.row(idx_1);
            m_curr_descriptors->emplace_back(des);

            keypoints_1.emplace_back(kp_1[idx_1].pt);
            keypoints_2.emplace_back(kp_2[idx_2].pt);

            final_matches.emplace_back(match[0]);
        }
    }

    std::cout << "[INFO]: Matched " << final_matches.size() << " points for triangulation." << std::endl;

    // display
    cv::Mat img_features;
    cv::drawMatches(img_1, kp_1, img_2, kp_2, final_matches, img_features);
    cv::imshow("Stereo", img_features);
}

void VisualOdometer::TriangulateKeypoints(
    const std::vector<cv::Point2f>& keypoints_1, 
    const std::vector<cv::Point2f>& keypoints_2, 
    std::vector<cv::Mat>& points)
{
    m_camera->Triangulate(keypoints_1, keypoints_2, points);
}

Eigen::Matrix4f VisualOdometer::CalcTransformation()
{
    cv::Mat curr_points_final;
    cv::Mat prev_points_final;

    // match points between curr and prev frames
    cv::Mat des_curr;
    cv::Mat des_prev;
    std::vector<std::vector<cv::DMatch>> matches;
    
    for (int i = 0; i < m_curr_descriptors->size(); i++)
    {
        des_curr.push_back((*m_curr_descriptors)[i]);
    }

    for (int i = 0; i < m_prev_descriptors->size(); i++)
    {
        des_prev.push_back((*m_prev_descriptors)[i]);
    }

    m_matcher->knnMatch(des_curr, des_prev, matches, 2);

    // conduct Lowe's ratio test
    const float dist_thres = 30.0;
    const float ratio_thres = 0.5;
    std::vector<cv::DMatch> final_matches;

    for (int i = 0; i < matches.size(); i++)
    {
        float d1 = matches[i][0].distance;
        float d2 = matches[i][1].distance;

        if (d1 < dist_thres && d1 / d2 < ratio_thres)
        {
            unsigned int idx_curr = matches[i][0].queryIdx;
            unsigned int idx_prev = matches[i][0].trainIdx;

            curr_points_final.push_back((*m_curr_points)[idx_curr]);
            prev_points_final.push_back((*m_prev_points)[idx_prev]);

            final_matches.emplace_back(matches[i][0]);
        }
    }

    std::cout << "[INFO]: Matched " << curr_points_final.rows << " points for tracking." << std::endl;

    cv::Mat img_features;
    cv::drawMatches(*m_curr_img, *m_curr_keypoints, *m_prev_img, *m_prev_keypoints, final_matches, img_features);
    cv::imshow("Temporal", img_features);

    // find transformation
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(4, 4);
    unsigned int N = curr_points_final.rows;

    if (N < 10)
    {
        m_success = false;
        return trans;
    }

    std::vector<PointPair> point_pairs;

    for (int i = 0; i < N; i++)
    {
        Eigen::Vector3f p1;
        Eigen::Vector3f p2;

        p1(0) = prev_points_final.at<float>(i, 0);
        p1(1) = prev_points_final.at<float>(i, 1);
        p1(2) = prev_points_final.at<float>(i, 2);
        p2(0) = curr_points_final.at<float>(i, 0);
        p2(1) = curr_points_final.at<float>(i, 1);
        p2(2) = curr_points_final.at<float>(i, 2);

        point_pairs.emplace_back(std::make_pair(p1, p2));
    }

    Eigen::Matrix4f mat_trans = m_solver->Solve(point_pairs);
    m_success = true;

    return mat_trans;
}
