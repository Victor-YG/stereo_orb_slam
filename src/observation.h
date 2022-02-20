#pragma once

#include <Eigen/Geometry>


struct Observation
{
    Observation(unsigned int point_id_, float u_l_, float v_l_, float u_r_, float v_r_, float sigma_):
        point_id(point_id_), u_l(u_l_), v_l(v_l_), u_r(u_r_), v_r(v_r_), sigma(sigma_) {}

    unsigned int    point_id;
    float           u_l;
    float           v_l;
    float           u_r;
    float           v_r;
    float           sigma;
};
