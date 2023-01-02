#include "camera_utils.h"

#include <sstream>
#include <fstream>
#include <iostream>


CameraModel::Stereo* LoadCamera(const std::string& file_path)
{
    std::ifstream stream(file_path, std::ifstream::in);

    if (!stream.is_open())
    {
        std::cout << "[FAIL]: Failed to open camera file." << std::endl;
        return nullptr;
    }

    std::string type;
    float fx_l, fy_l, cx_l, cy_l, k1_l, k2_l, p1_l, p2_l, k3_l;
    float fx_r, fy_r, cx_r, cy_r, k1_r, k2_r, p1_r, p2_r, k3_r;
    float b;
    float d_l[5], d_r[5];
    float pose_l[16], pose_r[16];

    char char_arr[512];
    while (stream.getline(char_arr, 512))
    {
        if (stream.bad() || stream.eof())
            break;

        std::string line(char_arr);
        int idx = line.find_first_of("=");
        std::string key = line.substr(0, idx);
        std::string value = line.substr(idx + 1, line.length());

        if (key == "type") type = value;
        else if (key == "fx_l") fx_l = stod(value);
        else if (key == "fy_l") fy_l = stod(value);
        else if (key == "cx_l") cx_l = stod(value);
        else if (key == "cy_l") cy_l = stod(value);
        else if (key == "fx_r") fx_r = stod(value);
        else if (key == "fy_r") fy_r = stod(value);
        else if (key == "cx_r") cx_r = stod(value);
        else if (key == "cy_r") cy_r = stod(value);
        else if (key == "b") b = stod(value);
        else if (key == "d_l") ReadDistortionCoef(value, d_l);
        else if (key == "d_r") ReadDistortionCoef(value, d_r);
        else if (key == "T_l") ReadTransformation(value, pose_l);
        else if (key == "T_r") ReadTransformation(value, pose_r);
        else std::cout << "[WARN]: Unrecognized key '" << key << "' found." << std::endl;
    }

    CameraModel::PinholeCamera cam_l(fx_l, fy_l, cx_l, cy_l, d_l, pose_l);
    CameraModel::PinholeCamera cam_r(fx_r, fy_r, cx_r, cy_r, d_r, pose_r);

    if (type == "StereoRectified" && fx_l == fy_l == fx_r == fy_r && cy_l == cy_r)
    {
        return new CameraModel::StereoRectified(cam_l, cam_r);
    }
    else
    {
        return new CameraModel::Stereo(cam_l, cam_r);
    }
}

void ReadDistortionCoef(const std::string& value, float coef[5])
{
    std::stringstream ss(value);

    for (int i = 0; i < 5; i++)
    {
        ss >> coef[i];
    }
}

void ReadTransformation(const std::string& value, float trans[16])
{
    std::stringstream ss(value);

    for (int i = 0; i < 16; i++)
    {
        ss >> trans[i];
    }
}
