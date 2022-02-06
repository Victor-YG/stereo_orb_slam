#pragma once

#include "map_point.h"
#include "camera_frame.h"

#include <array>
#include <vector>
#include <string>
#include <fstream>


void SavePointsToPLY(
    const std::string& file_path, 
    const std::vector<std::array<float, 3>>& points)
{
    std::ofstream output;
    output.open(file_path);

    if (!output.is_open()) return;

    int N = points.size();

    // write header
    output << "ply" << std::endl;
    output << "format ascii 1.0" << std::endl;
    output << "comment object: list of points" << std::endl;
    output << "element vertex " << N << std::endl;
    output << "property float x" << std::endl;
    output << "property float y" << std::endl;
    output << "property float z" << std::endl;
    output << "end_header" << std::endl;
    
    // write points
    for (int i = 0; i < N; i++)
    {
        output << points[i][0] << " ";
        output << points[i][1] << " ";
        output << points[i][2] << std::endl;
    }

    output.close();
}

void SavePosesToPLY(
    const std::string& file_path, 
    const std::vector<Frame>& frames)
{
    std::ofstream output;
    output.open(file_path);

    if (!output.is_open()) return;

    unsigned int N = frames.size();

    // write header
    output << "ply" << std::endl;
    output << "format ascii 1.0" << std::endl;
    output << "comment object: list of points" << std::endl;
    output << "element vertex " << N << std::endl;
    output << "property float x" << std::endl;
    output << "property float y" << std::endl;
    output << "property float z" << std::endl;
    output << "end_header" << std::endl;

    // write frames
    for (int i = 0; i < N; i++)
    {
        Eigen::Matrix4f pose = frames[i].GlobalPose();
        output << pose(0, 3) << " ";
        output << pose(1, 3) << " ";
        output << pose(2, 3) << std::endl;
    }
}

void SaveMapToPLY(
    const std::string& file_path, 
    const std::vector<Frame>& frames,
    const std::vector<MapPoint>& points)
{
    std::ofstream output;
    output.open(file_path);

    if (!output.is_open()) return;

    unsigned int num_frames = frames.size();
    unsigned int num_points = points.size();
    int N = num_frames + num_points;

    // write header
    output << "ply" << std::endl;
    output << "format ascii 1.0" << std::endl;
    output << "comment object: list of points" << std::endl;
    output << "element vertex " << N << std::endl;
    output << "property float x" << std::endl;
    output << "property float y" << std::endl;
    output << "property float z" << std::endl;
    output << "property uchar red" << std::endl;
    output << "property uchar green" << std::endl;
    output << "property uchar blue" << std::endl;
    output << "end_header" << std::endl;

    // write frames
    for (int i = 0; i < num_frames; i++)
    {
        Eigen::Matrix4f pose = frames[i].GlobalPose();
        output << pose(0, 3) << " ";
        output << pose(1, 3) << " ";
        output << pose(2, 3) << " ";
        output <<   0 << " ";
        output << 255 << " ";
        output <<   0 << std::endl;
    }

    // write points
    for (int i = 0; i < num_points; i++)
    {
        std::array<float, 3> position = points[i].Position();
        output << position[0] << " ";
        output << position[1] << " ";
        output << position[2] << " ";

        unsigned int num_observed = points[i].Descriptors().size();
        if (num_observed > 10)
        {   // Burnt Orange
            output << 204 << " ";
            output <<  85 << " ";
            output <<   0 << std::endl;
        }
        else if (num_observed > 5)
        {   // Bright Orange
            output << 255 << " ";
            output << 172 << " ";
            output <<  28 << std::endl;
        }
        else if (num_observed > 1)
        {   // yellow
            output << 255 << " ";
            output << 255 << " ";
            output <<   0 << std::endl;
        }
        else
        {   // white
            output << 255 << " ";
            output << 255 << " ";
            output << 255 << std::endl;
        }
    }

    output.close();
}
