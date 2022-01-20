#pragma once

#include <string>
#include <vector>
#include <algorithm>

typedef std::pair<std::string, std::string> ImagePair;

void LoadDatasetKitti(const std::string& folder, std::vector<ImagePair>& frames);
void LoadDatasetEuRoc(const std::string& folder, std::vector<ImagePair>& frames);