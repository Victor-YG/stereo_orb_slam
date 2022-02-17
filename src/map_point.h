#pragma once

#include <array>

#include <Eigen/Geometry>
#include "opencv4/opencv2/opencv.hpp"


class MapPoint
{
public:
    MapPoint(float x, float y, float z)
    {
        m_position[0] = x;
        m_position[1] = y;
        m_position[2] = z;
    }

    void Transform(Eigen::Matrix4f trans)
    {
        Eigen::Vector4f position(m_position[0], m_position[1], m_position[2], 1.0);
        position = trans * position;
        m_position[0] = position(0);
        m_position[1] = position(1);
        m_position[2] = position(2);
    }

    void Position(std::array<double, 3> pos)
    {
        m_position[0] = pos[0];
        m_position[1] = pos[1];
        m_position[2] = pos[2];
    }
    
    void AddDescriptor(cv::Mat descriptor) { m_descriptors.emplace_back(descriptor); }

    std::array<float, 3> Position() const { return m_position; }
    std::vector<cv::Mat> Descriptors() const { return m_descriptors; }

private:
    std::array<float, 3> m_position;
    std::vector<cv::Mat> m_descriptors;
};
