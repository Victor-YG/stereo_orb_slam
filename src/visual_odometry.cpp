#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "dataset.h"
#include "ply_utils.h"
#include "camera_utils.h"
#include "camera_model.h"
#include "visual_odometer.h"

#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "sophus/se3.hpp"
#include "gflags/gflags.h"
#include "opencv4/opencv2/opencv.hpp"


typedef std::chrono::high_resolution_clock Timer;

// input variables
DEFINE_string(dataset, "", "Dataset name. e.g. kitti, EuRoc etc.");
DEFINE_string(folder, "", "Data folder.");
DEFINE_string(camera, "", "Stereo camera information file.");

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
    // return 0;

    // load camera
    CameraModel::Stereo* camera = LoadCamera(FLAGS_camera);

    // setup odometer
    VisualOdometer vo = VisualOdometer();
    vo.Camera(camera);

    // main loop
    int start = 0;
    int N = frames.size();
    // int N = 100;

    namedWindow("Stereo", cv::WINDOW_AUTOSIZE);
    namedWindow("Temporal", cv::WINDOW_AUTOSIZE);

    std::vector<Eigen::Matrix4f> poses;
    std::vector<std::array<float, 3>> waypoints;
    Eigen::Matrix4f curr_pose = Eigen::Matrix4f::Identity(4, 4);
    
    for (int i = start; i < N; i += 1)
    {
        std::cout << "[INFO]: frame #" << i << ": " << frames[i].first << std::endl;

        // read images
        cv::Mat img_l = cv::imread(frames[i].first, cv::IMREAD_GRAYSCALE);
        cv::Mat img_r = cv::imread(frames[i].second, cv::IMREAD_GRAYSCALE);

        // // undistort images
        // cv::Mat img_l_undistorted;
        // cv::Mat img_r_undistorted;
        // cv::Mat cam_mat = camera->m_cam_1.GetCameraMatrix();
        // cv::Mat dist_coef = camera->m_cam_1.GetDistortionCoef();
        // // std::cout << cam_mat << std::endl;
        // // std::cout << dist_coef << std::endl;

        // cv::undistort(img_l, img_l_undistorted, camera->m_cam_1.GetCameraMatrix(), camera->m_cam_1.GetDistortionCoef());
        // cv::undistort(img_r, img_r_undistorted, camera->m_cam_2.GetCameraMatrix(), camera->m_cam_2.GetDistortionCoef());

        // track
        cv::waitKey(10);
        auto t1 = Timer::now();

        curr_pose.block<3, 3>(0, 0) = Eigen::Quaternionf(curr_pose.block<3, 3>(0, 0)).normalized().toRotationMatrix();
        Sophus::SE3 pose_se3(curr_pose);
        Eigen::Matrix4f trans = vo.Track(img_l, img_r);
        // Eigen::Matrix4f trans = vo.Track(img_l_undistorted, img_r_undistorted);
        Sophus::SE3 trans_se3(trans);
        curr_pose = pose_se3.matrix();

        // pose_se3 = trans_se3 * pose_se3;
        curr_pose = curr_pose * trans;
        Eigen::Vector3f position = pose_se3.translation();
        
        // std::array<float, 3> waypoint = {
        //     position(0), 
        //     position(1), 
        //     position(2)
        // };

        std::array<float, 3> waypoint = {
            curr_pose(0, 3), 
            curr_pose(1, 3), 
            curr_pose(2, 3)
        };

        poses.emplace_back(curr_pose);
        waypoints.emplace_back(waypoint);

        auto t2 = Timer::now();
        std::cout << "[INFO]: Elapsed " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;

        std::cout << "[INFO]: tran = " << std::endl;
        std::cout << trans << std::endl;
        std::cout << "[INFO]: pose = " << std::endl;
        std::cout << curr_pose << std::endl;

        // break;
    }
    
    cv::destroyAllWindows();
    std::cout << "[INFO]: End of sequence." << std::endl;

    SavePointsToPLY("./waypoints.ply", waypoints);
    return 0;
}
