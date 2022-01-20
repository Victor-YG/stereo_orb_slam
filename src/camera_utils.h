#pragma once

#include "camera_model.h"

#include <string>


CameraModel::Stereo* LoadCamera(const std::string& file_path);
void ReadDistortionCoef(const std::string& value, float coef[5]);
void ReadTransformation(const std::string& value, float trans[16]);