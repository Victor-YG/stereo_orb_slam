#include "dataset.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>


void LoadDatasetKitti(const std::string& folder, std::vector<ImagePair>& frames)
{
    frames.clear();

    std::vector<std::string> frames_l;
    std::vector<std::string> frames_r;

    const std::string img_path_l = folder + "/image_0";
    const std::string img_path_r = folder + "/image_1";

    const std::filesystem::path path_l{img_path_l};
    const std::filesystem::path path_r{img_path_r};

    for (auto const& image : std::filesystem::directory_iterator{path_l})
    {
        frames_l.emplace_back(image.path());
    }
    for (auto const& image : std::filesystem::directory_iterator{path_r})
    {
        frames_r.emplace_back(image.path());
    }

    assert(frames_l.size() == frames_r.size());

    std::sort(frames_l.begin(), frames_l.end());
    std::sort(frames_r.begin(), frames_r.end());

    for (int i = 0; i < frames_l.size(); i++)
    {
        frames.emplace_back(std::make_pair(frames_l[i], frames_r[i]));
    }
}

void LoadDatasetEuRoc(const std::string& folder, std::vector<ImagePair>& frames)
{
    frames.clear();

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
            const std::string img_path_r = folder + "/mav0/cam1/data/" + frames_r[i].second;
            frames.emplace_back(std::make_pair(img_path_l, img_path_r));
        }
        else
        {
            std::cout << "[WARN]: mismatch in time stamp found." << std::endl;
        }
    }
}

void LoadDatasetOther(const std::string& folder, std::vector<ImagePair>& frames)
{
    frames.clear();

    std::vector<std::string> frames_l;
    std::vector<std::string> frames_r;

    const std::filesystem::path path{folder};
    for (auto const& f : std::filesystem::directory_iterator{path})
    {
        std::string filepath = f.path();
        if (filepath.find("l.png") != std::string::npos)
        {
            frames_l.emplace_back(filepath);
        }
        if (filepath.find("r.png") != std::string::npos)
        {
            frames_r.emplace_back(filepath);
        }
    }

    assert(frames_l.size() == frames_r.size());

    std::sort(frames_l.begin(), frames_l.end());
    std::sort(frames_r.begin(), frames_r.end());

    for (int i = 0; i < frames_l.size(); i++)
    {
        frames.emplace_back(std::make_pair(frames_l[i], frames_r[i]));
    }
}