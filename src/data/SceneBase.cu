/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */
#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"

#include "SceneBase.h"


static __global__ void PointInAnyImage(StaticDeviceTensor<double, 2> point, StaticDeviceTensor<double, 2> rotation,
                                       StaticDeviceTensor<double, 2> translation,
                                       StaticDeviceTensor<double, 2> intrinsics, StaticDeviceTensor<float, 1> out_mask,
                                       int h, int w)
{
    int image_id = blockIdx.x;

    Quat q       = ((Quat*)&rotation(image_id, 0))[0];
    Vec3 t       = ((Vec3*)&translation(image_id, 0))[0];
    Vec5 k_coeff = ((Vec5*)&intrinsics(0, 0))[0];
    IntrinsicsPinholed K(k_coeff);
    K.s = 0;

    SE3 T = SE3(q, t).inverse();

    for (int point_id = threadIdx.x; point_id < point.sizes[0]; point_id += blockDim.x)
    {
        Vec3 position = ((Vec3*)&point(point_id, 0))[0];

        Vec3 view_pos = T * position;

        Vec2 image_pos = K.project(view_pos);

        ivec2 pixel = image_pos.array().round().cast<int>();

        if (pixel(0) >= 0 && pixel(0) < w && pixel(1) >= 0 && pixel(1) < h)
        {
            out_mask(point_id) = 1;
        }
        // ((Vec3*)&out_point(tid, 0))[0] = new_pos;
    }
}


torch::Tensor SceneBase::PointInAnyImage(torch::Tensor points)
{
    auto linear_points = points.reshape({-1, 3}).to(torch::kDouble);

    auto mask = torch::zeros({linear_points.size(0)}, points.options().dtype(torch::kFloat32));

    ::PointInAnyImage<<<frames.size(), 128>>>(linear_points, pose->rotation, pose->translation,
                                              camera_model->intrinsics, mask, camera_model->h, camera_model->w);
    CUDA_SYNC_CHECK_ERROR();

    auto out_size = points.sizes().vec();
    out_size.pop_back();

    mask = mask.reshape(out_size);


    return mask;
}
