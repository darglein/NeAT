/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once

#include "HelperStructs.h"

#include <saiga/core/geometry/aabb.h>


inline Eigen::Vector<int, -1> Delinearize(int i, Eigen::Vector<int, -1> size, bool swap_xy)
{
    int D = size.rows();
    Eigen::Vector<int, -1> res;
    res.resize(D);

    int tmp = i;
    if (swap_xy)
    {
        for (int d = D - 1; d >= 0; --d)
        {
            res(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for (int d = 0; d < D; ++d)
        {
            res(d) = tmp % size(d);
            tmp /= size(d);
        }
    }

    return res;
}

HD inline std::tuple<bool, float, float> IntersectBoxRayPrecise(float* pos_min, float* pos_max, float* origin,
                                                                float* direction, int D)
{
    double t_near = -236436436;  // maximums defined in float.h
    double t_far  = 43637575;

    for (int i = 0; i < D; i++)
    {  // we test slabs in every direction
        if (direction[i] == 0)
        {  // ray parallel to planes in this direction
            if ((origin[i] < pos_min[i]) || (origin[i] >= pos_max[i]))
            {
                return {false, 0, 0};  // parallel AND outside box : no intersection possible
            }
        }
        else
        {  // ray not parallel to planes in this direction
            float T_1 = (pos_min[i] - origin[i]) / direction[i];
            float T_2 = (pos_max[i] - origin[i]) / direction[i];

            if (T_1 > T_2)
            {  // we want T_1 to hold values for intersection with near plane
                // std::swap(T_1, T_2);
                auto tmp = T_1;
                T_1      = T_2;
                T_2      = tmp;
            }
            if (T_1 > t_near)
            {
                t_near = T_1;
            }
            if (T_2 < t_far)
            {
                t_far = T_2;
            }
            if ((t_near > t_far) || (t_far < 0))
            {
                return {false, 0, 0};
            }
        }
    }
    return {true, t_near, t_far};
}

HD inline bool BoxContainsPoint(float* pos_min, float* pos_max, float* p, int D)
{
    for (int d = 0; d < D; ++d)
    {
        if (pos_min[d] > p[d] || pos_max[d] < p[d])
        {
            return false;
        }
    }
    return true;
}


class HyperTreeBaseImpl : public virtual torch::nn::Module, public torch::nn::Cloneable<HyperTreeBaseImpl>
{
   public:
    HyperTreeBaseImpl(int d, int max_depth);
    virtual ~HyperTreeBaseImpl() {}

    void CloneInto(HyperTreeBaseImpl* other);


    NodeBatchedSamples GroupSamplesPerNodeGPU(const SampleList& samples, int group_size);

    // For each inactive node we compute the output features as if they had been sampled normally.
    // Used to optimize network after a structure update to a reasonable initial solution.
    //
    // Input:
    //    active_grid: float [num_nodes, num_features, x, y, z]
    // Output:
    //    interpolated_grid: float [num_nodes, num_features, x, y, z]
    torch::Tensor InterpolateGridForInactiveNodes(torch::Tensor active_grid);

    torch::Tensor GetNodePositionScaleForId(torch::Tensor node_ids)
    {
        CHECK(node_position_min.defined());
        CHECK(node_scale.defined());
        auto pos = torch::index_select(node_position_min, 0, node_ids);
        auto sca = torch::index_select(node_scale, 0, node_ids).unsqueeze(1);
        return torch::cat({pos, sca}, 1);
    }

    torch::Tensor ActiveNodeTensor() { return active_node_ids; }

    // The input are the samples batched by node id!
    // global_samples
    //      float [num_groups, group_size, D]
    // node_indices
    //      long [num_groups]
    torch::Tensor ComputeLocalSamples(torch::Tensor global_samples, torch::Tensor node_indices);

    // Traverse the tree and finds the active node id containing the sample position.
    // The returned masked indicates if this sample is valid (i.e. the sample is inside an active node).
    // The mask is should be only 0 if the containing node has been culled or the sample is outside of [-1,1]
    //
    // Input
    //      global_samples float [..., D]
    // Return
    //      node_id, long [N]
    //      mask,    float [N]
    //
    std::tuple<torch::Tensor, torch::Tensor> NodeIdForPositionGPU(torch::Tensor global_samples);

    // For each grid element x we check if x+epsilon is in a different node.
    // If yes, this is a edge-element and we generate 2 output samples using the same coordinates.
    //
    // This should be used to greate a edge-regularizer that makes neighboring nodes have the same value at the edge.
    //
    SampleList NodeNeighborSamples(Eigen::Vector<int, -1> size, double epsilon);

    SampleList UniformPhantomSamplesGPU(Eigen::Vector<int, -1> size, bool swap_xy);

    SampleList CreateSamplesForRays(const RayList& rays, int max_samples_per_node, bool interval_jitter);


    // If all children of a node are culled this node will be also culled
    void UpdateCulling();

    // Given a node id, this function creates uniform sample locations inside this node (in global space)
    //
    // Input
    //      node_id int [N]
    // Return
    //      position, float [N, grid_size, grid_size, grid_size, D]
    torch::Tensor UniformGlobalSamples(torch::Tensor node_id, int grid_size);

    void SetErrorForActiveNodes(torch::Tensor error, std::string strategy = "override");

    void SplitNode(int to_split) {}
    void ResetLoss() {}

    void FinalizeExp(double alpha) {}

    // Sets all nodes of layer i to active.
    // All others to inactive
    void SetActive(int depth);
    void UpdateActive();

    torch::Tensor GlobalNodeIdToLocalActiveId(torch::Tensor node_id)
    {
        auto res = torch::index_select(node_active_prefix_sum, 0, node_id.reshape({-1}));
        return res.reshape(node_id.sizes());
    }

    // Return
    //      long [num_inactive_nodes]
    torch::Tensor InactiveNodeIds();

    std::vector<AABB> ActiveNodeBoxes();

    virtual void reset();

    torch::Device device() { return node_parent.device(); }
    int NumNodes() { return node_parent.size(0); }
    int NumActiveNodes() { return active_node_ids.size(0); }
    int D() { return node_position_min.size(1); }
    int NS() { return node_children.size(1); }

    template <int U>
    friend struct DeviceHyperTree;

    // int [num_nodes]
    // -1 for the root node
    torch::Tensor node_parent;

    // int [num_nodes, NS]
    // -1 for leaf nodes
    torch::Tensor node_children;

    // float [num_nodes, D]
    torch::Tensor node_position_min, node_position_max;

    // float [num_nodes]
    torch::Tensor node_scale;

    // float [num_nodes] (invalid error == -1)
    torch::Tensor node_error;

    // float [num_nodes]
    torch::Tensor node_max_density;

    // int [num_nodes]
    torch::Tensor node_depth;

    // int [num_nodes] boolean 0/1
    torch::Tensor node_active;

    // int [num_nodes] boolean 0/1 default all 0
    torch::Tensor node_culled;

    // Exclusive prefix sum of 'node_active'
    // long [num_nodes]
    torch::Tensor node_active_prefix_sum;

    // long [num_active_nodes]
    torch::Tensor active_node_ids;
};

TORCH_MODULE(HyperTreeBase);