#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "dataset.h"
#include "ply_utils.h"
#include "pose_graph.h"
#include "math_utils.h"
#include "camera_utils.h"
#include "camera_model.h"
#include "loop_detector.h"
#include "visual_odometer.h"
#include "bundle_adjuster.h"
#include "reprojection_error.h"

#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "gflags/gflags.h"
#include "opencv4/opencv2/opencv.hpp"


typedef std::chrono::high_resolution_clock Timer;

// input variables
DEFINE_string(dataset, "", "Dataset name. e.g. kitti, EuRoc etc.");
DEFINE_string(folder, "", "Data folder.");
DEFINE_string(camera, "", "Stereo camera information file.");
DEFINE_int32(refine_interval, 10, "Refinement interval for local BA.");
DEFINE_string(output_suffix, "slam", "Suffix to describe output file.");


void InitializeStereoReprojectionError(const cv::Mat& projection_l, const cv::Mat& projection_r);

int main(int argc, char** argv)
{
    // read inputs
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_dataset.empty())
    {
        std::cerr << "[FAIL]: Please provide the dataset name using -dataset.\n";
        return -1;
    }

    if (FLAGS_folder.empty())
    {
        std::cerr << "[FAIL]: Please provide the path to dataset using -folder.\n";
        return -1;
    }

    if (FLAGS_camera.empty()) {
        std::cerr << "[FAIL]: Please provide information file for stereo camera using -camera.\n";
        return -1;
    }

    // load images
    std::vector<ImagePair> frames;
    if      (FLAGS_dataset == "kitti") LoadDatasetKitti(FLAGS_folder, frames);
    else if (FLAGS_dataset == "EuRoc") LoadDatasetEuRoc(FLAGS_folder, frames);
    else
    {
        std::cerr << "[FAIL]: Unknown dataset '" << FLAGS_dataset << "' provided.\n";
        return -1;
    }

    // load camera
    CameraModel::Stereo* camera = LoadCamera(FLAGS_camera);

    // set up reprojection error
    CameraModel::PinholeCamera* cam_l = camera->GetCamera1();
    CameraModel::PinholeCamera* cam_r = camera->GetCamera2();
    cv::Mat projection_l = cam_l->GetProjectionMatrix();
    cv::Mat projection_r = cam_r->GetProjectionMatrix();
    InitializeStereoReprojectionError(projection_l, projection_r);

    // create data containers
    std::vector<Frame*>    cam_frames;
    std::vector<MapPoint*> ldm_points;
    std::vector<PoseGraphEdge>  edges;
    Eigen::Matrix4f curr_pose = Eigen::Matrix4f::Identity();

    // setup odometer
    VisualOdometer vo(cam_frames, ldm_points);
    vo.Camera(camera);

    // setup bundle Adjuster
    BundleAdjuster ba(cam_frames, ldm_points);

    // setup pose graph optimizer
    PoseGraphOptimizer po(ba, cam_frames, edges);

    // setup loop detector
    LoopDetector ld(po, cam_frames, edges);
    ld.LoadVocabulary("../res/vocabulary/small_db.yml.gz");

    // main loop
    int start = 0;
    int N = frames.size();
    // N = 10;

    for (int i = start; i < N; i += 1)
    {
        // read images
        std::cout << "[INFO]: frame #" << i << ": " << frames[i].first << std::endl;
        cv::Mat img_l = cv::imread(frames[i].first, cv::IMREAD_GRAYSCALE);
        cv::Mat img_r = cv::imread(frames[i].second, cv::IMREAD_GRAYSCALE);

        cv::waitKey(10);
        auto t1 = Timer::now();

        // track
        Eigen::Matrix4f trans = vo.Track(img_l, img_r);

        // local BA
        unsigned int n = cam_frames.size();
        if (n != 0 && n % FLAGS_refine_interval == 0)
        {
            ba.Optimize(std::max(0, int(n - 2 * FLAGS_refine_interval)), n);
        }

        FrameDataContainer* frame_data = vo.GetCurrFrameData();
        ld.Query(frame_data->descriptors);
        ld.Track(frame_data->descriptors);

        auto t2 = Timer::now();
        std::cout << "[INFO]: Elapsed " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;

        // update pose
        curr_pose = curr_pose * trans;
        Normalize(curr_pose);

        std::cout << "[INFO]: tran = " << std::endl;
        std::cout << trans << std::endl;
        std::cout << "[INFO]: pose = " << std::endl;
        std::cout << curr_pose << std::endl;
    }

    std::cout << "[INFO]: End of sequence." << std::endl;

    // final ba
    ba.Optimize(0, cam_frames.size() - 1);

    // save results
    std::string suffix = FLAGS_output_suffix;
    std::string waypoint_filename = "./waypoints_" + suffix + ".ply";
    std::string map_filename = "./map_" + suffix + ".ply";
    std::string pose_graph_filename = "./pose_graph_" + suffix + ".ply";
    SavePosesToPLY(waypoint_filename, cam_frames);
    SaveMapToPLY(map_filename, cam_frames, ldm_points);
    SavePoseGraphToPLY(pose_graph_filename, cam_frames, edges);

    // save vocabulary
    ld.SaveVocabulary("../res/vocabulary/small_db.yml.gz");

    return 0;
}

void InitializeStereoReprojectionError(const cv::Mat& projection_l, const cv::Mat& projection_r)
{
    std::array<double, 12> p_l;
    std::array<double, 12> p_r;

    p_l[ 0] = (double) projection_l.at<float>(0, 0);
    p_l[ 1] = (double) projection_l.at<float>(0, 1);
    p_l[ 2] = (double) projection_l.at<float>(0, 2);
    p_l[ 3] = (double) projection_l.at<float>(0, 3);
    p_l[ 4] = (double) projection_l.at<float>(1, 0);
    p_l[ 5] = (double) projection_l.at<float>(1, 1);
    p_l[ 6] = (double) projection_l.at<float>(1, 2);
    p_l[ 7] = (double) projection_l.at<float>(1, 3);
    p_l[ 8] = (double) projection_l.at<float>(2, 0);
    p_l[ 9] = (double) projection_l.at<float>(2, 1);
    p_l[10] = (double) projection_l.at<float>(2, 2);
    p_l[11] = (double) projection_l.at<float>(2, 3);

    p_r[ 0] = (double) projection_r.at<float>(0, 0);
    p_r[ 1] = (double) projection_r.at<float>(0, 1);
    p_r[ 2] = (double) projection_r.at<float>(0, 2);
    p_r[ 3] = (double) projection_r.at<float>(0, 3);
    p_r[ 4] = (double) projection_r.at<float>(1, 0);
    p_r[ 5] = (double) projection_r.at<float>(1, 1);
    p_r[ 6] = (double) projection_r.at<float>(1, 2);
    p_r[ 7] = (double) projection_r.at<float>(1, 3);
    p_r[ 8] = (double) projection_r.at<float>(2, 0);
    p_r[ 9] = (double) projection_r.at<float>(2, 1);
    p_r[10] = (double) projection_r.at<float>(2, 2);
    p_r[11] = (double) projection_r.at<float>(2, 3);

    ReprojectionError::SetLeftProjection(p_l);
    ReprojectionError::SetRightProjection(p_r);
}
