/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "geometry_ex_ex.h"

#include "modules/IndirectGridSample3D.h"
GeometryExEx::GeometryExEx(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params)
    : HierarchicalNeuralGeometry(num_channels, D, params, tree)
{
    std::cout << "Type: ExEx" << std::endl;
    std::vector<long> features_grid_shape;
    for (int i = 0; i < D; ++i)
    {
        features_grid_shape.push_back(params->net_params.grid_size);
    }

    grid_sampler = NeuralGridSampler(false, true);
    register_module("grid_sampler", grid_sampler);

    {
        // only keep active nodes in memory
        explicit_grid_generator =
            ExplicitFeatureGrid(params->octree_params.tree_optimizer_params.max_active_nodes,
                                params->net_params.grid_features, features_grid_shape, params->train_params.grid_init);
    }

    register_module("explicit_grid_generator", explicit_grid_generator);

    std::cout << "Feature Grid: " << TensorInfo(explicit_grid_generator->grid_data) << std::endl;
    std::cout << "Numel: " << explicit_grid_generator->grid_data.numel()
              << " Memory: " << explicit_grid_generator->grid_data.numel() * sizeof(float) / 1000000.0 << " MB"
              << std::endl;

    std::cout << "=== ============= ===" << ConsoleColor::RESET << std::endl;
}

void GeometryExEx::AddParametersToOptimizer()
{
    HierarchicalNeuralGeometry::AddParametersToOptimizer();


    std::cout << "Optimizing Explicit Grid with (ADAM) LR " << params->train_params.lr_exex_grid_adam << std::endl;
    optimizer_adam->add_param_group(
        {explicit_grid_generator->parameters(),
         std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_exex_grid_adam)});
}


torch::Tensor GeometryExEx::SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    if (global_coordinate.numel() == 0)
    {
        // No samples -> just return and empty tensor
        return global_coordinate;
    }


    torch::Tensor grid, local_samples, neural_features;

    {
        SAIGA_OPTIONAL_TIME_MEASURE("explicit_grid_generator", timer);
        // [num_nodes, num_features, 11, 11, 11]
        auto local_node_id = tree->GlobalNodeIdToLocalActiveId(node_id);
        grid               = explicit_grid_generator->forward(local_node_id);
    }

    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeLocalSamples", timer);
        local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
    }
    CHECK_EQ(local_samples.requires_grad(), global_coordinate.requires_grad());

    {
        SAIGA_OPTIONAL_TIME_MEASURE("grid_sampler->forward", timer);
        // [num_groups, group_size, num_features]
        neural_features = grid_sampler->forward(grid, local_samples);
    }

    return neural_features;
}


torch::Tensor GeometryExEx::VolumeRegularizer()
{
    torch::Tensor grid;
    {
        grid = explicit_grid_generator(tree->GlobalNodeIdToLocalActiveId(tree->ActiveNodeTensor()));
    }

    torch::Tensor tv_loss, edge_loss, zero_loss;

    if (params->train_params.loss_tv > 0)
    {
        torch::Tensor tv_grid;
        tv_grid = grid;

        torch::Tensor factor;
        TVLoss tv;
        tv_loss = tv.forward(tv_grid, factor) * params->train_params.loss_tv;
    }


    if (params->train_params.loss_edge > 0)
    {
        Eigen::Vector<int, -1> shape_v;
        shape_v.resize(D);
        for (int i = 0; i < D; ++i)
        {
            shape_v(i) = params->net_params.grid_size;
        }

        SampleList neighbor_samples = tree->NodeNeighborSamples(shape_v, 0.001);
        int num_rays                = neighbor_samples.size();

        torch::Tensor neural_features;
        auto local_samples = tree->ComputeLocalSamples(neighbor_samples.global_coordinate, neighbor_samples.node_id);
        {
            auto local_node_id = tree->GlobalNodeIdToLocalActiveId(neighbor_samples.node_id);
            neural_features    = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
        }
        auto ray_index   = neighbor_samples.ray_index;
        auto per_ray_sum = torch::zeros({num_rays, neural_features.size(1)}, neural_features.options());
        per_ray_sum.index_add_(0, ray_index, neural_features);
        // [num_rays, channels]
        auto t1 = per_ray_sum.slice(0, 0, per_ray_sum.size(0), 2);
        auto t2 = per_ray_sum.slice(0, 1, per_ray_sum.size(0), 2);

        // [num_rays]
        auto edge_error = (t1 - t2).abs().mean(1);
        edge_error *= torch::rand_like(edge_error);
        edge_loss = edge_error.mean() * params->train_params.loss_edge * neural_features.size(1);
    }
    torch::Tensor loss;
    if (edge_loss.defined())
    {
        if (loss.defined())
            loss += edge_loss;
        else
            loss = edge_loss;
    }
    if (tv_loss.defined())
    {
        if (loss.defined())
            loss += tv_loss;
        else
            loss = tv_loss;
    }
    if (zero_loss.defined())
    {
        if (loss.defined())
            loss += zero_loss;
        else
            loss = zero_loss;
    }
    // exit(0);
    return loss;
}
torch::Tensor GeometryExEx::SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    torch::Tensor local_samples, neural_features, density;

    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeLocalSamples", timer);
        local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("IndirectGridSample3D", timer);
        auto local_node_id = tree->GlobalNodeIdToLocalActiveId(node_id);
        neural_features    = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("DecodeFeatures", timer);
        density = DecodeFeatures(neural_features);
    }

    return density;
}

void GeometryExEx::InterpolateInactiveNodes(HyperTreeBase old_tree)
{
    torch::NoGradGuard ngg;
    ScopedTimerPrintLine tim("InterpolateInactiveNodes");
    // [num_inactive_nodes]
    auto new_active_nodes = tree->active_node_ids;

    CUDA_SYNC_CHECK_ERROR();
    auto global_coordinate = tree->UniformGlobalSamples(new_active_nodes, params->net_params.grid_size);
    auto global_coordinate_block =
        global_coordinate.reshape({global_coordinate.size(0), -1, global_coordinate.size(-1)});
    global_coordinate_block = global_coordinate_block.reshape({-1, 3});
    CUDA_SYNC_CHECK_ERROR();
    auto [node_id, mask] = old_tree->NodeIdForPositionGPU(global_coordinate_block);
    auto local_samples   = old_tree->ComputeLocalSamples(global_coordinate_block, node_id);
    CUDA_SYNC_CHECK_ERROR();
    torch::Tensor neural_features;
    {
        auto local_node_id = old_tree->GlobalNodeIdToLocalActiveId(node_id);
        neural_features    = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
    }
    CUDA_SYNC_CHECK_ERROR();
    neural_features        = neural_features.reshape({global_coordinate.size(0), global_coordinate.size(1),
                                                      global_coordinate.size(2), global_coordinate.size(3), -1});
    neural_features        = neural_features.permute({0, 4, 1, 2, 3});
    auto new_local_node_id = tree->GlobalNodeIdToLocalActiveId(new_active_nodes);
    explicit_grid_generator->grid_data.index_copy_(0, new_local_node_id, neural_features);
    CUDA_SYNC_CHECK_ERROR();
}