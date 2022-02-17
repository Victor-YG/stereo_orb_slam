#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "dataset.h"
#include "ply_utils.h"
#include "math_utils.h"
#include "camera_utils.h"
#include "camera_model.h"
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
DEFINE_int32(refine_interval, 5, "Refinement interval for local BA.");
DEFINE_string(output_suffix, "_slam", "Suffix to describe output file.");


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
    CameraModel::PinholeCamera* cam = camera->GetCamera1();
    cv::Mat mat_projection = cam->GetProjectionMatrix();
    // TODO::to directly use the projection matrix which is more generic (work for right camera as well)
    float fx = mat_projection.at<float>(0, 0);
    float fy = mat_projection.at<float>(1, 1);
    float cx = mat_projection.at<float>(0, 2);
    float cy = mat_projection.at<float>(1, 2);
    ReprojectionError::SetCameraParams(fx, fy, cx, cy);

    // create data containers
    std::vector<Frame*>   cam_frames;
    std::vector<MapPoint> ldm_points;
    Eigen::Matrix4f curr_pose = Eigen::Matrix4f::Identity();

    // setup odometer
    VisualOdometer vo = VisualOdometer(cam_frames, ldm_points);
    vo.Camera(camera);

    // setup bundle Adjuster
    BundleAdjuster ba = BundleAdjuster(cam_frames, ldm_points);

    // main loop
    int start = 0;
    int N = frames.size();

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

    // save results
    std::string suffix = FLAGS_output_suffix;
    std::string waypoint_filename = "./waypoints" + suffix + ".ply";
    std::string map_filename = "./map" + suffix + ".ply";
    SavePosesToPLY(waypoint_filename, cam_frames);
    SaveMapToPLY(map_filename, cam_frames, ldm_points);

    return 0;
}
