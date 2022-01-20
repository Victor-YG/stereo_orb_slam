#pragma Once

#include <array>
#include <vector>
#include <string>
#include <fstream>


void SavePointsToPLY(std::string file_path, std::vector<std::array<float, 3>> points)
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
}