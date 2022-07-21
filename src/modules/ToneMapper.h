/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/camera/HDR.h"
#include "saiga/core/util/ini/Params.h"

#include <torch/torch.h>

struct PhotometricCalibrationParams : public ParamsBase
{
  SAIGA_PARAM_STRUCT(PhotometricCalibrationParams);
  SAIGA_PARAM_STRUCT_FUNCTIONS;

  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        SAIGA_PARAM(response_enable);
        SAIGA_PARAM(response_range);
        SAIGA_PARAM(response_lr);

        SAIGA_PARAM(exposure_enable);
        SAIGA_PARAM(exposure_mult);
        SAIGA_PARAM(exposure_lr);


        SAIGA_PARAM(sensor_bias_enable);
        SAIGA_PARAM(sensor_bias_size_w);
        SAIGA_PARAM(sensor_bias_lr);
    }

    bool response_enable = true;
    float response_range = 2;
    float response_lr    = 0.1;

    bool exposure_enable = true;
    bool exposure_mult   = false;
    float exposure_lr    = 0.01;

    // the size in h will be computed from the aspect ratio
    bool sensor_bias_enable = true;
    int sensor_bias_size_w  = 32;
    float sensor_bias_lr    = 0.05;
};


class CameraResponseNetImpl : public torch::nn::Module
{
   public:
    CameraResponseNetImpl(int params, int num_channels, float initial_gamma, float size = 1,
                          float leaky_clamp_value = 0);

    std::vector<Saiga::DiscreteResponseFunction<float>> GetCRF();

    void ApplyConstraints() {}

    torch::Tensor forward(torch::Tensor image);

    torch::Tensor ParamLoss();

    int NumParameters() { return response.size(3); }

    torch::nn::functional::GridSampleFuncOptions options;
    torch::Tensor response;
    torch::Tensor leaky_value;
    float range;
};

TORCH_MODULE(CameraResponseNet);

class PhotometricCalibrationImpl : public torch::nn::Module
{
   public:
    PhotometricCalibrationImpl(int num_images, int num_cameras, int h, int w, PhotometricCalibrationParams params);

    // Input:
    //      ray_integral, float [num_channels, num_pixels]
    //      uv,           float [num_pixels, 2] Range [0,1] !!!
    //      image_id,     float [num_pixels]
    // Output:
    //     image_intensity, float [num_channels, num_pixels]
    torch::Tensor forward(torch::Tensor ray_integral, torch::Tensor uv, torch::Tensor image_id);

    torch::Tensor ParameterLoss(torch::Tensor active_images);

    // Should be called after optimizer->step
    void ApplyConstraints();

    // both: float [num_images]
    torch::Tensor exposure_factor;
    torch::Tensor exposure_bias;

    torch::Tensor sensor_bias;

    CameraResponseNet response = nullptr;

    PhotometricCalibrationParams params;
};
TORCH_MODULE(PhotometricCalibration);
