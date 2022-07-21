/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once

#include "saiga/vision/torch/EigenTensor.h"
#include "saiga/vision/torch/ImageTensor.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
// #include <torch/extension.h>
#include "saiga/core/math/random.h"

#include <torch/torch.h>

using namespace Saiga;


struct PixelList
{
    // The image from which this pixel comes from
    // int [N]
    torch::Tensor image_id;

    // The camera for this ray (one camera can capture multiple images)
    // int [N]
    torch::Tensor camera_id;

    // uv coordinates in range [0, 1]
    // float [N, 2]
    torch::Tensor uv;

    // The pixel intensity sampled at the location 'uv'
    // float [num_channels, N]
    torch::Tensor target;

    // for each target pixel a factor that is multiplied to the loss
    // float [1, N]
    torch::Tensor target_mask;


    PixelList() {}

    // Stacks all rays into a single list
    PixelList(const std::vector<PixelList>& list)
    {
        std::vector<torch::Tensor> id_list;
        std::vector<torch::Tensor> ci_list;
        std::vector<torch::Tensor> uv_list;
        std::vector<torch::Tensor> tg_list;
        std::vector<torch::Tensor> tm_list;
        for (auto& l : list)
        {
            id_list.push_back(l.image_id);
            ci_list.push_back(l.camera_id);
            uv_list.push_back(l.uv);
            tg_list.push_back(l.target);
            if (l.target_mask.defined())
            {
                tm_list.push_back(l.target_mask);
            }
        }
        image_id  = torch::cat(id_list, 0);
        camera_id = torch::cat(ci_list, 0);
        uv        = torch::cat(uv_list, 0);
        target    = torch::cat(tg_list, 1);
        if (!tm_list.empty())
        {
            target_mask = torch::cat(tm_list, 1);
        }
    }

    void to(torch::Device device)
    {
        image_id  = image_id.to(device);
        camera_id = camera_id.to(device);
        uv        = uv.to(device);
        target    = target.to(device);
        if (target_mask.defined())
        {
            target_mask = target_mask.to(device);
        }
    }
};

// This struct contains a list of rays as tensors.
struct RayList
{
    // float [num_rays, D]
    torch::Tensor origin;

    // float [num_rays, D]
    torch::Tensor direction;

    RayList() {}

    // Stacks all rays into a single list
    RayList(const std::vector<RayList>& list)
    {
        std::vector<torch::Tensor> origin_list;
        std::vector<torch::Tensor> direction_list;
        for (auto& l : list)
        {
            origin_list.push_back(l.origin);
            direction_list.push_back(l.direction);
        }
        origin    = torch::cat(origin_list, 0);
        direction = torch::cat(direction_list, 0);
    }

    void Allocate(int num_rays, int D)
    {
        origin    = torch::empty({num_rays, D});
        direction = torch::empty({num_rays, D});
    }

    void to(torch::Device device)
    {
        // if(linear_pixel_location.defined()) linear_pixel_location = linear_pixel_location.to(device);
        origin    = origin.to(device);
        direction = direction.to(device);
        // if (pixel_uv.defined()) pixel_uv = pixel_uv.to(device);
    }

    RayList SubSample(torch::Tensor index)
    {
        RayList result;
        // result.linear_pixel_location = torch::index_select(linear_pixel_location, 0, index);
        result.origin    = torch::index_select(origin, 0, index);
        result.direction = torch::index_select(direction, 0, index);
        return result;
    }

    size_t Memory()
    {
        return direction.numel() * sizeof(float) + origin.numel() * sizeof(float);
        // +linear_pixel_location.numel() * sizeof(long);
    }

    int size() const { return origin.size(0); }

    int Dim() const { return origin.size(1); }

    template <int D>
    std::pair<Eigen::Vector<float, D>, Eigen::Vector<float, D>> GetRay(int i) const
    {
        CHECK_LT(i, size());
        CHECK_EQ(D, Dim());
        Eigen::Vector<float, D> o;
        Eigen::Vector<float, D> d;
        for (int k = 0; k < D; ++k)
        {
            o(k) = origin.template data_ptr<float>()[i * origin.stride(0) + k * origin.stride(1)];
            d(k) = direction.template data_ptr<float>()[i * direction.stride(0) + k * direction.stride(1)];
        }

        return {o, d};
    }
};


// A list of sample positions in the hyper tree stored as tensors.
struct SampleList
{
    int max_samples_per_ray = 0;

    // float [num_samples, D]
    torch::Tensor global_coordinate;

    // float [num_samples]
    torch::Tensor ray_t;

    // float [num_samples]
    torch::Tensor weight;

    // long [num_samples]
    torch::Tensor ray_index;

    // used for order dependent image formation
    // long [num_samples]
    torch::Tensor local_index_in_ray;

    // long [num_samples]
    torch::Tensor node_id;

    void to(torch::Device device)
    {
        global_coordinate  = global_coordinate.to(device);
        weight             = weight.to(device);
        ray_index          = ray_index.to(device);
        local_index_in_ray = local_index_in_ray.to(device);
        node_id            = node_id.to(device);
        ray_t              = ray_t.to(device);
    }

    int size() const { return global_coordinate.size(0); }

    void Allocate(int num_samples, int D, torch::Device device = torch::kCPU)
    {
        global_coordinate  = torch::empty({num_samples, D}, torch::TensorOptions(torch::kFloat32).device(device));
        weight             = torch::empty({num_samples}, torch::TensorOptions(torch::kFloat32).device(device));
        ray_t              = torch::empty({num_samples}, torch::TensorOptions(torch::kFloat32).device(device));
        ray_index          = torch::empty({num_samples}, torch::TensorOptions(torch::kLong).device(device));
        local_index_in_ray = torch::empty({num_samples}, torch::TensorOptions(torch::kLong).device(device));
        node_id            = torch::empty({num_samples}, torch::TensorOptions(torch::kLong).device(device));
    }

    SampleList Slice(int start, int end)
    {
        SampleList result;
        result.global_coordinate  = global_coordinate.slice(0, start, end);
        result.weight             = weight.slice(0, start, end);
        result.ray_index          = ray_index.slice(0, start, end);
        result.ray_t              = ray_t.slice(0, start, end);
        result.local_index_in_ray = local_index_in_ray.slice(0, start, end);
        result.node_id            = node_id.slice(0, start, end);
        return result;
    }

    void Shrink(int new_size)
    {
        global_coordinate  = global_coordinate.slice(0, 0, new_size);
        weight             = weight.slice(0, 0, new_size);
        ray_t              = ray_t.slice(0, 0, new_size);
        ray_index          = ray_index.slice(0, 0, new_size);
        local_index_in_ray = local_index_in_ray.slice(0, 0, new_size);
        node_id            = node_id.slice(0, 0, new_size);
    }
};

struct NodeBatchedSamples
{
    // float [num_groups, group_size, D]
    torch::Tensor global_coordinate;

    // float [num_groups, group_size, 1]
    torch::Tensor integration_weight;

    // 1 for valid samples, 0 for padded samples
    // float [num_groups, group_size, 1]
    torch::Tensor mask;

    // long [num_groups, group_size, 1]
    torch::Tensor ray_index;

    // Only used if the sample ordering is important
    // long [num_groups, group_size]
    torch::Tensor sample_index_in_ray;

    // int [num_groups]
    torch::Tensor node_ids;


    int GroupSize() { return global_coordinate.size(1); }

    void to(torch::Device device)
    {
        //        gt               = gt.to(device);
        //        local_samples  = local_samples.to(device);
        global_coordinate  = global_coordinate.to(device);
        integration_weight = integration_weight.to(device);
        mask               = mask.to(device);
        // node_position  = node_position.to(device);
        node_ids = node_ids.to(device);

        //        active_node_ids = active_node_ids.to(device);

        if (sample_index_in_ray.defined()) sample_index_in_ray = sample_index_in_ray.to(device);
        if (ray_index.defined()) ray_index = ray_index.to(device);
    }
};
