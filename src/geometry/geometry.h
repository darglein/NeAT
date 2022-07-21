/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/imgui_cuda.h"

#include "ImplicitNet.h"
#include "Settings.h"
#include "data/SceneBase.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

class NeuralGeometry : public torch::nn::Module
{
   public:
    NeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params)
        : num_channels(num_channels), D(D), params(params)
    {
    }

    virtual void train(int epoch_id, bool on)
    {
        torch::nn::Module::train(on);
        c10::cuda::CUDACachingAllocator::emptyCache();
        if (on)
        {
            if (!optimizer_adam && !optimizer_sgd)
            {
                CreateGeometryOptimizer();
            }
            if (optimizer_adam) optimizer_adam->zero_grad();
            if (optimizer_sgd) optimizer_sgd->zero_grad();
            if (optimizer_rms) optimizer_rms->zero_grad();
            if (optimizer_decoder) optimizer_decoder->zero_grad();
        }
    }

    void ResetGeometryOptimizer() { CreateGeometryOptimizer(); }

    void CreateGeometryOptimizer()
    {
        optimizer_adam =
            std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));

        optimizer_rms = std::make_shared<torch::optim::RMSprop>(std::vector<torch::Tensor>(),
                                                                torch::optim::RMSpropOptions().lr(10));

        optimizer_sgd = std::make_shared<torch::optim::SGD>(std::vector<torch::Tensor>(), torch::optim::SGDOptions(10));
        AddParametersToOptimizer();
    }



    virtual void PrintInfo() {}
    virtual void PrintGradInfo(int epoch_id, TensorBoardLogger* logger) {}


    void OptimizerStep(int epoch_id)
    {
        if (optimizer_sgd)
        {
            optimizer_sgd->step();
            optimizer_sgd->zero_grad();
        }
        if (optimizer_adam)
        {
            optimizer_adam->step();
            optimizer_adam->zero_grad();
        }
        if (optimizer_rms)
        {
            optimizer_rms->step();
            optimizer_rms->zero_grad();
        }
        if (optimizer_decoder)
        {
            optimizer_decoder->step();
            optimizer_decoder->zero_grad();
        }
    }

    void UpdateLearningRate(double factor)
    {
        if (optimizer_adam) UpdateLR(optimizer_adam.get(), factor);
        if (optimizer_sgd) UpdateLR(optimizer_sgd.get(), factor);
        if (optimizer_rms) UpdateLR(optimizer_rms.get(), factor);
        if (optimizer_decoder) UpdateLR(optimizer_decoder.get(), factor);
    }


    // Compute the 'simple' integral by just adding each sample value to the given ray index.
    // The ordering of the samples is not considered.
    //
    // Computes:
    //      sample_integral[ray_index[i]] += sample_value[i]
    //
    // Input:
    //      sample_value [num_groups, group_size, num_channels]
    //      ray_index [N]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesXRay(torch::Tensor sample_values, torch::Tensor integration_weight,
                                       torch::Tensor ray_index, int num_channels, int num_rays);


    // Blends the samples front-to-back using alpha blending. This is used for a RGB-camera model (non xray) and the
    // implementation follows the raw2outputs function of NeRF. However, in our case it is more complicated because
    // each ray can have a different number of samples. The computation is done in the following steps:
    //
    //  1. Sort the sample_values into a matrix of shape: [num_rays, max_samples_per_ray, num_channels]
    //     Each row, is also ordered correctly in a front to back fashion. If a ray has less than max_samples_per_ray
    //     samples, the remaining elements are filled with zero.
    //
    //
    // Input:
    //      sample_value [any_shape, num_channels]
    //      ray_index [any_shape]
    //
    //      // The local id of each sample in the ray. This is used for sorting!
    //      sample_index_in_ray [any_shape]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesAlphaBlending(torch::Tensor sample_values, torch::Tensor integration_weight,
                                                torch::Tensor ray_index, torch::Tensor sample_index_in_ray,
                                                int num_channels, int num_rays, int max_samples_per_ray);



   protected:
    std::shared_ptr<torch::optim::Adam> optimizer_decoder;

    std::shared_ptr<torch::optim::Adam> optimizer_adam;
    std::shared_ptr<torch::optim::SGD> optimizer_sgd;
    std::shared_ptr<torch::optim::RMSprop> optimizer_rms;

    int num_channels;
    int D;
    std::shared_ptr<CombinedParams> params;

    virtual void AddParametersToOptimizer() = 0;
};

class HierarchicalNeuralGeometry : public NeuralGeometry
{
   public:
    HierarchicalNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params, HyperTreeBase tree);

    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const NodeBatchedSamples& combined_samples,
                                                                        torch::Tensor per_ray_loss);
    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const SampleList& combined_samples,
                                                                        torch::Tensor per_ray_loss);

    virtual torch::Tensor VolumeRegularizer() { return torch::Tensor(); }


    HyperTreeBase tree = nullptr;

    torch::Tensor ComputeImage(SampleList all_samples, int num_channels, int num_pixels);

    // Input:
    //   global_coordinate [num_samples, D]
    //   node_id [num_samples]
    virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) { return {}; }

    // Output:
    //      value [num_groups, group_size, num_channels]
    torch::Tensor SampleVolumeBatched(torch::Tensor global_coordinate, torch::Tensor sample_mask, torch::Tensor node_id)
    {
        CHECK_EQ(global_coordinate.dim(), 3);
        if (global_coordinate.numel() == 0)
        {
            return torch::empty({global_coordinate.size(0), global_coordinate.size(1), num_channels},
                                global_coordinate.options());
        }

        torch::Tensor neural_features, density;
        {
            neural_features = SampleVolume(global_coordinate, node_id);
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("DecodeFeatures", timer);
            density = DecodeFeatures(neural_features);
        }
        CHECK_EQ(density.dim(), sample_mask.dim());
        density = density * sample_mask;
        return density;
    }



    virtual void to(torch::Device device, bool non_blocking = false) override
    {
        NeuralGeometry::to(device, non_blocking);
    }

    // Returns [volume_density, volume_node_index, volume_valid]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniformSampledVolume(std::vector<long> shape,
                                                                                 int num_channels);

    void SaveVolume(TensorBoardLogger* tblogger, std::string tb_name, std::string out_dir, int num_channels,
                    float intensity_scale, int size);

    static FCBlock shared_decoder;
    FCBlock decoder                        = nullptr;


    // Evaluates the octree at the inactive-node's feature positions and sets the respective feature vectors.
    // This should be called before changing the octree structure, because then some inactive nodes will become active.
    // The newly active nodes will have a good initialization after this method.
    virtual void InterpolateInactiveNodes(HyperTreeBase old_tree) {}


   protected:
    // Takes the sample locations (and the corresponding tree-node-ids) and retrieves the values from the
    // hierarchical data structure. The input samples must be 'grouped' by the corresponding node-id.
    // The per-sample weight is multiplied to the raw sample output.
    //
    // Input:
    //      global_coordinate [num_groups, group_size, 3]
    //      weight            [num_groups, group_size]
    //      node_id           [num_groups]
    //
    // Output:
    //      value [num_groups, group_size, num_features]
    virtual torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) = 0;



    virtual void AddParametersToOptimizer();

    // Input:
    //      neural_features [num_groups, group_size, num_channels]
    torch::Tensor DecodeFeatures(torch::Tensor neural_features);

   public:
    Saiga::CUDA::CudaTimerSystem* timer = nullptr;
};
