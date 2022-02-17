#pragma once

#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"


struct ReprojectionError
{
    ReprojectionError(double u_, double v_):
        u(u_), v(v_) {}
    
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
        const T x_n = p[0] / p[2];
        const T y_n = p[1] / p[2];

        const T u_hat = fx * x_n + cx;
        const T v_hat = fy * y_n + cy;

        residuals[0] = u_hat - u;
        residuals[1] = v_hat - v;

        return true;
    }

    static void SetCameraParams(double fx_, double fy_, double cx_, double cy_)
    {
        fx = fx_;
        fy = fy_;
        cx = cx_;
        cy = cy_;
    }

    static ceres::CostFunction* Create(
        const double observed_x, 
        const double observed_y)
    {
        return (
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(observed_x, observed_y)
            )
        );
    }

    // projection params
    inline static double fx;
    inline static double fy;
    inline static double cx;
    inline static double cy;

    // observation
    double u;
    double v;
};
