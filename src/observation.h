#pragma once

#include <Eigen/Geometry>


struct Observation
{
    Observation(unsigned int point_id_, float u_, float v_, float sigma_):
        point_id(point_id_), u(u_), v(v_), sigma(sigma_) {}

    unsigned int    point_id;
    float           u;
    float           v;
    float           sigma;
};
