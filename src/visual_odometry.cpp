#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "gflags/gflags.h"
#include "opencv4/opencv2/opencv.hpp"


typedef std::pair<std::string, std::string> ImagePair;

// input variables
DEFINE_string(dataset, "", "Data folder.");
DEFINE_string(camera, "", "Stereo camera information file.");


void LoadDataset(const std::string& folder, std::vector<ImagePair>& frames);

int main(int argc, char** argv)
{
    // read inputs
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_dataset.empty())
    {
        std::cerr << "[FAIL]: Please provide the path to dataset using -dataset.\n";
        return -1;
    }

    if (FLAGS_camera.empty()) {
        std::cerr << "[FAIL]: Please provide information file for stereo camera using -camera.\n";
        return -1;
    }

    std::vector<ImagePair> frames;
    LoadDataset(FLAGS_dataset, frames);

    // main loop
    int N = frames.size();
    for (int i = 0; i < N; i++)
    {
        cv::Mat img_l = cv::imread(frames[i].first, cv::IMREAD_GRAYSCALE);
        cv::Mat img_r = cv::imread(frames[i].second, cv::IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> keypoints_1;
        std::vector<cv::KeyPoint> keypoints_2;
        std::vector<cv::DMatch> matches;

        cv::Mat img_matched_features;
        cv::drawMatches(img_l, keypoints_1, img_r, keypoints_2, matches, img_matched_features);
        cv::imshow("Frame (" + std::to_string(i) + " / " + std::to_string(N) + ")", img_matched_features);

        // sleep(0.05);
        cv::waitKey(500);
        cv::destroyAllWindows();
    }

    return 0;
}

void LoadDataset(const std::string& folder, std::vector<ImagePair>& frames)
{
    const std::string csv_l = folder + "/mav0/cam0/data.csv";
    const std::string csv_r = folder + "/mav0/cam1/data.csv";

    char char_arr[512];
    std::vector<std::pair<std::string, std::string>> frames_l;
    std::vector<std::pair<std::string, std::string>> frames_r;

    // read left camera csv
    std::ifstream stream_l(csv_l, std::fstream::in);
    if (!stream_l.is_open())
    {
        std::cout << "[FAIL]: Failed to open " << csv_l << std::endl;
        return;
    }

    stream_l.getline(char_arr, 512); // skip first line
    while (stream_l.getline(char_arr, 512))
    {
        if (stream_l.bad() || stream_l.eof())
            break;

        std::string line(char_arr);
        int idx = line.find_first_of(",");

        std::string time_stamp = line.substr(0, idx);
        std::string image_name = line.substr(idx + 1, line.length());
        image_name.erase(std::remove_if(image_name.begin(), image_name.end(), ::isspace), image_name.end());
        frames_l.emplace_back(std::make_pair(time_stamp, image_name));
    }

    // read right camera csv
    std::ifstream stream_r(csv_r, std::fstream::in);
    if (!stream_r.is_open())
    {
        std::cout << "[FAIL]: Failed to open " << csv_r << std::endl;
        return;
    }

    stream_r.getline(char_arr, 512); // skip first line
    while (stream_r.getline(char_arr, 512))
    {
        if (stream_r.bad() || stream_r.eof())
            break;

        std::string line(char_arr);
        int idx = line.find_first_of(",");

        std::string time_stamp = line.substr(0, idx);
        std::string image_name = line.substr(idx + 1, line.length());
        image_name.erase(std::remove_if(image_name.begin(), image_name.end(), ::isspace), image_name.end());
        frames_r.emplace_back(std::make_pair(time_stamp, image_name));
    }

    // match images with same time stamp
    for (int i = 0; i < frames_l.size(); i++)
    {
        if (frames_l[i].first == frames_r[i].first)
        {
            const std::string img_path_l = folder + "/mav0/cam0/data/" + frames_l[i].second;
            const std::string img_path_r = folder + "/mav0/cam0/data/" + frames_r[i].second;
            frames.emplace_back(std::make_pair(img_path_l, img_path_r));
        }
        else
        {
            std::cout << "[WARN]: mismatch in time stamp found." << std::endl;
        }
    }
}