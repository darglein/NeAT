/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <torch/torch.h>

namespace Saiga
{
//    uv: float [N, 2]
inline torch::Tensor UVToPixel(torch::Tensor uv, int w, int h, bool align_corners)
{
    auto u = uv.slice(1, 0, 1);
    auto v = uv.slice(1, 1, 2);
    if (align_corners)
    {
        u = u * (w - 1);
        v = v * (h - 1);
    }
    else
    {
        u = u * w - 0.5f;
        v = v * h - 0.5f;
    }

    return torch::cat({u, v}, 1);
}

inline torch::Tensor PixelToUV(torch::Tensor px, int w, int h, bool align_corners)
{
    auto x = px.slice(1, 0, 1);
    auto y = px.slice(1, 1, 2);
    if (align_corners)
    {
        x = x / (w - 1);
        y = y / (h - 1);
    }
    else
    {
        x = (x + 0.5f) / w;
        y = (y + 0.5f) / h;
    }

    return torch::cat({x, y}, 1);
}

class CameraPoseModuleImpl : public torch::nn::Module
{
   public:
    using PoseType = Saiga::SE3;

    CameraPoseModuleImpl(Saiga::ArrayView<PoseType> poses);

    void ApplyTangent();


    void AddNoise(double noise_translation, double noise_rotation);


    torch::Tensor RotatePoint(torch::Tensor p, torch::Tensor index);



    // double: [N, 4]
    torch::Tensor rotation;
    // double : [ N, 3 ]
    torch::Tensor rotation_tangent;

    // double: [N, 3]
    torch::Tensor translation;
};
TORCH_MODULE(CameraPoseModule);

enum class CameraModelType
{
    PINHOLE            = 0,
    PINHOLE_DISTORTION = 1,
    OCAM               = 1,
};

class CameraModelModuleImpl : public torch::nn::Module
{
   public:
    // Basic pinhole without distortion
    CameraModelModuleImpl(int h, int w, Saiga::ArrayView<IntrinsicsPinholed> data);


    // Projects pixel coordinates from image space into camera space.
    // output = ((ip(0) - cx) / fx, (ip(1) - cy) / fy, 1);
    //
    // Input:
    //    camera_index int [N]
    //    pixel_coords float [N,2]
    //    depth float [N]
    //
    // Output:
    //    coords float [N, 3]
    torch::Tensor Unproject(torch::Tensor camera_index, torch::Tensor pixel_coords, torch::Tensor depth);

    std::vector<IntrinsicsPinholed> DownloadPinhole();


    CameraModelType type;

    int h, w;


    // [num_cameras, num_model_params]
    // Pinhole:              [num_cameras, 5]
    // Pinhole + Distortion: [num_cameras, 5 + 8]
    // OCam:                 [num_cameras, 5 + world2cam_coefficients]
    torch::Tensor intrinsics;


    void AddNoise(double noise_k)
    {
        if (noise_k > 0)
        {
            torch::NoGradGuard ngg;
            std::cout << "Adding Intrinsics Noise" << std::endl;
            auto noise = (torch::rand_like(intrinsics) - 0.5) * noise_k;
            PrintTensorInfo(noise);
            intrinsics.add_(noise);
        }
    }
};
TORCH_MODULE(CameraModelModule);

}  // namespace Saiga