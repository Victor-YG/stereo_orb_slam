#pragma once

#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"


struct ReprojectionError
{
    ReprojectionError(double u_l_, double v_l_, double u_r_, double v_r_):
        u_l(u_l_), v_l(v_l_), u_r(u_r_), v_r(v_r_) {}
    
    template <typename T>
    bool operator()(
        const T* const camera, 
        const T* const point, 
        T* residuals) const
    {
        // transform to camera frame of reference
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // projection
        const T s_l_hat = 1.0 / (p_l[8] * p[0] + p_l[9] * p[1] + p_l[10] * p[2] + p_l[11] * 1.0);
        const T u_l_hat = (p_l[0] * p[0] + p_l[1] * p[1] + p_l[2] * p[2] + p_l[3] * 1.0) * s_l_hat;
        const T v_l_hat = (p_l[4] * p[0] + p_l[5] * p[1] + p_l[6] * p[2] + p_l[7] * 1.0) * s_l_hat;

        const T s_r_hat = 1.0 / (p_r[8] * p[0] + p_r[9] * p[1] + p_r[10] * p[2] + p_r[11] * 1.0);
        const T u_r_hat = (p_r[0] * p[0] + p_r[1] * p[1] + p_r[2] * p[2] + p_r[3] * 1.0) * s_r_hat;
        const T v_r_hat = (p_r[4] * p[0] + p_r[5] * p[1] + p_r[6] * p[2] + p_r[7] * 1.0) * s_r_hat;

        residuals[0] = u_l_hat - u_l;
        residuals[1] = v_l_hat - v_l;
        residuals[2] = u_r_hat - u_r;
        residuals[3] = v_r_hat - v_r;

        return true;
    }

    static void SetLeftProjection(const std::array<double, 12>& projection_l)
    {
        p_l = projection_l;
    }

    static void SetRightProjection(const std::array<double, 12>& projection_r)
    {
        p_r = projection_r;
    }

    static ceres::CostFunction* Create(
        const double u_l, const double v_l, 
        const double u_r, const double v_r)
    {
        return (
            new ceres::AutoDiffCostFunction<ReprojectionError, 4, 6, 3>(
                new ReprojectionError(u_l, v_l, u_r, v_r)
            )
        );
    }

    // projection params
    inline static std::array<double, 12> p_l; // left projection matrix
    inline static std::array<double, 12> p_r; // right projection matrix

    // observation
    double u_l;
    double v_l;
    double u_r;
    double v_r;
};
