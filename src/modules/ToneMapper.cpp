/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "ToneMapper.h"

#include "saiga/vision/torch/ImageTensor.h"
using namespace Saiga;


CameraResponseNetImpl::CameraResponseNetImpl(int params, int num_channels, float initial_gamma, float range,
                                             float leaky_clamp_value)
    : range(range)
{
    Saiga::DiscreteResponseFunction<float> crf;
    crf = Saiga::DiscreteResponseFunction<float>(params);
    crf.MakeGamma(initial_gamma);
    crf.normalize(range);

    options.align_corners(true);
    options.padding_mode(torch::kBorder);
    options.mode(torch::kBilinear);

    response = torch::from_blob(crf.irradiance.data(), {1, 1, 1, (long)crf.irradiance.size()}, torch::kFloat).clone();

    // repeat across channels
    response = response.repeat({1, num_channels, 1, 1});

    if (leaky_clamp_value > 0)
    {
        leaky_value = torch::empty({1}, torch::kFloat).fill_(leaky_clamp_value);
    }

    register_parameter("response", response);
}
torch::Tensor CameraResponseNetImpl::forward(torch::Tensor input_image)
{
    auto image = input_image;
    if (image.dim() == 2)
    {
        image = image.unsqueeze(0).unsqueeze(2);
    }
    SAIGA_ASSERT(image.dtype() == response.dtype());
    SAIGA_ASSERT(image.dim() == 4);
    SAIGA_ASSERT(image.size(1) == response.size(1));

    torch::Tensor leak_add;
    if (this->is_training() && leaky_value.defined())
    {
        leaky_value = leaky_value.to(image.device());
        // torch::Tensor clamp_low  = image < 0;
        torch::Tensor clamp_high = image > range;

        // below 0 leak
        // leak_add = (image * leaky_value) * clamp_low;

        // above 1 leak
        leak_add = (image - range) * clamp_high;
    }


    int num_batches  = image.size(0);
    int num_channels = image.size(1);

    auto batched_response = response.repeat({num_batches, 1, 1, 1});

    // the input is [0, range]
    image = image * (1.f / range);


    // The grid sample uv space is from -1 to +1
    image = image * 2.f - 1.f;

    // Add zero-y coordinate because gridsample is only implemented for 2D and 3D
    auto yoffset = torch::zeros_like(image);
    auto x       = torch::cat({image.unsqueeze(4), yoffset.unsqueeze(4)}, 4);

    auto result = torch::ones_like(image);
    for (int i = 0; i < num_channels; ++i)
    {
        // Slice away the channel dimension
        auto sl                   = x.slice(1, i, i + 1).squeeze(1);
        auto response_sl          = batched_response.slice(1, i, i + 1);
        result.slice(1, i, i + 1) = torch::nn::functional::grid_sample(response_sl, sl, options);
    }

    if (leak_add.defined())
    {
        result += leak_add;
    }

    return result.reshape(input_image.sizes());
}

torch::Tensor CameraResponseNetImpl::ParamLoss()
{
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));

    auto low = response.slice(3, 0, NumParameters() - 2);
    auto up  = response.slice(3, 2, NumParameters());

    torch::Tensor target = response.clone();  // torch::empty_like(response);

    // Set first value to zero and last value to 1
    target.slice(3, 0, 1).zero_();
    // target.slice(3, NumParameters() - 1, NumParameters()).fill_(1);

    // Set middle values to mean of neighbouring values
    // -> Force smoothness
    target.slice(3, 1, NumParameters() - 1) = (up + low) * 0.5f;

    double smoothness_factor = 1e-5;
    double factor            = NumParameters() * sqrt(smoothness_factor);
    return torch::mse_loss(response * factor, target * factor, torch::Reduction::Sum);
}
std::vector<Saiga::DiscreteResponseFunction<float>> CameraResponseNetImpl::GetCRF()
{
    auto r     = response.cpu().to(torch::kFloat32).contiguous();
    float* ptr = r.data_ptr<float>();


    std::vector<Saiga::DiscreteResponseFunction<float>> crfs;

    for (int c = 0; c < 3; ++c)
    {
        Saiga::DiscreteResponseFunction<float> crf(NumParameters());
        for (int i = 0; i < NumParameters(); ++i)
        {
            crf.irradiance[i] = ptr[i + c * NumParameters()];
        }
        crfs.push_back(crf);
    }
    return crfs;
}


PhotometricCalibrationImpl::PhotometricCalibrationImpl(int num_images, int num_cameras, int h, int w,
                                                       PhotometricCalibrationParams params)
    : params(params)
{
    CHECK_GT(num_images, 0);
    CHECK_GT(num_cameras, 0);
    CHECK_GT(w, 0);
    CHECK_GT(h, 0);

    exposure_factor = torch::ones({num_images});
    //    exposure_bias   = torch::full({num_images}, params.bias_init);
    exposure_bias = torch::full({num_images}, 0.);
    register_parameter("exposure_factor", exposure_factor);
    register_parameter("exposure_bias", exposure_bias);

    int size_w = params.sensor_bias_size_w;
    int size_h = iCeil(params.sensor_bias_size_w * (float(h) / w));
    std::cout << "Sensor bias map " << size_w << "x" << size_h << std::endl;
    sensor_bias = torch::full({1, 1, size_h, size_w}, 0.);
    register_parameter("sensor_bias", sensor_bias);

    if (params.response_enable)
    {
        response = CameraResponseNet(25, 1, 1, params.response_range, 1);
        register_module("response", response);
    }
}

torch::Tensor PhotometricCalibrationImpl::forward(torch::Tensor ray_integral, torch::Tensor uv, torch::Tensor image_id)
{
    if (params.exposure_enable)
    {
        auto exposure_factor_selected = torch::index_select(exposure_factor, 0, image_id);
        auto exposure_bias_selected   = torch::index_select(exposure_bias, 0, image_id);

        if (params.exposure_mult)
        {
            ray_integral = ray_integral * exposure_factor_selected;
        }
        ray_integral = ray_integral + exposure_bias_selected;
    }

    if (params.sensor_bias_enable)
    {
        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(false);

        auto uv_samples = (uv * 2 - 1).unsqueeze(0).unsqueeze(0);

        // [1, channels, 1, num_samples]
        auto sensor_bias_add = torch::nn::functional::grid_sample(sensor_bias, uv_samples, opt);

        // [channels, num_samples]
        sensor_bias_add = sensor_bias_add.squeeze(0).squeeze(1);

        ray_integral += sensor_bias_add;
    }


    if (response)
    {
        ray_integral = response->forward(ray_integral);
    }

    return ray_integral;
}

torch::Tensor PhotometricCalibrationImpl::ParameterLoss(torch::Tensor active_images)
{
    torch::Tensor param_loss = torch::zeros({1}, exposure_factor.device());
    if (response)
    {
        param_loss += response->ParamLoss();
    }

    return param_loss;
}

void PhotometricCalibrationImpl::ApplyConstraints()
{
    torch::NoGradGuard ngg;
    // Make sure the bias is always positive!
    exposure_bias.clamp_min_(0);
    sensor_bias.clamp_min_(0);
}
