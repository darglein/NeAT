/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "HyperTree.h"



HyperTreeBaseImpl::HyperTreeBaseImpl(int d, int max_depth)
{
    int NS        = 1 << d;
    int num_nodes = 0;
    // NS^0 + NS^1 + NS^2 + ... + NS^max_depth
    // NS^(depth+1)-1
    for (int i = 0; i <= max_depth; ++i)
    {
        long n = 1;
        for (int j = 0; j < i; ++j)
        {
            n *= NS;
        }
        num_nodes += n;
    }

    node_parent       = torch::empty({num_nodes}, torch::kInt32);
    node_children     = -torch::ones({num_nodes, NS}, torch::kInt32);
    node_position_min = torch::empty({num_nodes, d}, torch::kFloat32);
    node_position_max = torch::empty({num_nodes, d}, torch::kFloat32);
    node_scale        = torch::empty({num_nodes}, torch::kFloat32);
    //    node_diagonal_length   = torch::empty({num_nodes}, torch::kFloat32);

    // -1 means not computed yet
    node_error             = -torch::ones({num_nodes}, torch::kFloat32);
    node_max_density       = -torch::ones({num_nodes}, torch::kFloat32);

    node_active            = torch::zeros({num_nodes}, torch::kInt32);
    node_culled            = torch::zeros({num_nodes}, torch::kInt32);
    node_depth             = torch::zeros({num_nodes}, torch::kInt32);
    node_active_prefix_sum = torch::zeros({num_nodes}, torch::kInt32);

    register_buffer("node_parent", node_parent);
    register_buffer("node_children", node_children);
    register_buffer("node_position_min", node_position_min);
    register_buffer("node_position_max", node_position_max);
    register_buffer("node_scale", node_scale);
    register_buffer("node_error", node_error);
    register_buffer("node_max_density", node_max_density);
    register_buffer("node_active", node_active);
    register_buffer("node_culled", node_culled);
    register_buffer("node_depth", node_depth);
    register_buffer("node_active_id", node_active_prefix_sum);

    active_node_ids = torch::zeros({1}, torch::kLong);
    SetActive(0);
    register_buffer("active_node_ids", active_node_ids);

    using Vec = Eigen::Matrix<float, -1, 1>;

    auto get_node = [&](int node_id) -> std::pair<Vec, Vec>
    {
        Vec pos_min, pos_max;
        pos_min.resize(d);
        pos_max.resize(d);
        for (int i = 0; i < d; ++i)
        {
            pos_min(i) = node_position_min.data_ptr<float>()[node_id * d + i];
            pos_max(i) = node_position_max.data_ptr<float>()[node_id * d + i];
        }
        return {pos_min, pos_max};
    };

    auto set_node = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth)
    {
        float scale = ((float)depth / (max_depth)) * 2 - 1;
        if (max_depth == 0) scale = 0;

        //        float diag = (pos_max - pos_min).norm();
        for (int i = 0; i < d; ++i)
        {
            node_position_min.data_ptr<float>()[node_id * d + i] = pos_min(i);
            node_position_max.data_ptr<float>()[node_id * d + i] = pos_max(i);
        }
        node_depth.data_ptr<int>()[node_id]   = depth;
        node_parent.data_ptr<int>()[node_id]  = parent;
        node_scale.data_ptr<float>()[node_id] = scale;
        // node_diagonal_length.data_ptr<float>()[node_id] = diag;
    };

    int current_node = 0;
    // nodes[current_node++] = Node(-1, 0, -Vec::Ones(), false, Vec::Ones());
#if 1
    int start_layer = 0;
    set_node(current_node++, -1, -Vec(d).setOnes(), Vec(d).setOnes(), 0);
    for (int depth = 1; depth <= max_depth; ++depth)
    {
        int n = current_node - start_layer;
        for (int j = 0; j < n; ++j)
        {
            int node_id = start_layer + j;

            auto [pos_min, pos_max] = get_node(node_id);
            auto center             = (pos_min + pos_max) * 0.5;

            for (int i = 0; i < NS; ++i)
            {
                Vec new_min(d);
                Vec new_max(d);

                for (int k = 0; k < d; ++k)
                {
                    if ((i >> k & 1) == 0)
                    {
                        new_min[k] = pos_min[k];
                        new_max[k] = center[k];
                    }
                    else
                    {
                        new_min[k] = center[k];
                        new_max[k] = pos_max[k];
                    }
                }
                node_children.data_ptr<int>()[node_id * NS + i] = current_node;
                set_node(current_node++, node_id, new_min, new_max, depth);
            }
        }
        start_layer = current_node - n * NS;
    }
#endif
    std::cout << "> Max Nodes2: " << num_nodes << std::endl;
}

void HyperTreeBaseImpl::CloneInto(HyperTreeBaseImpl* other)
{
    other->to(this->device());
    auto my_buffers    = this->buffers();
    auto other_buffers = other->buffers();
    for (int i = 0; i < my_buffers.size(); ++i)
    {
        other_buffers[i].resize_(my_buffers[i].sizes());
        other_buffers[i].copy_(my_buffers[i].clone());
    }
}

void HyperTreeBaseImpl::reset()
{
    register_buffer("node_parent", node_parent);
    register_buffer("node_children", node_children);
    register_buffer("node_position_min", node_position_min);
    register_buffer("node_position_max", node_position_max);
    register_buffer("node_scale", node_scale);
    register_buffer("node_error", node_error);
    register_buffer("node_active", node_active);
    register_buffer("node_culled", node_culled);
    register_buffer("node_depth", node_depth);
    register_buffer("node_active_id", node_active_prefix_sum);

    active_node_ids = torch::zeros({1}, torch::kLong);
    register_buffer("active_node_ids", active_node_ids);
}

void HyperTreeBaseImpl::SetActive(int depth)
{
    node_active.set_data((node_depth == depth).to(torch::kInt32));
    // PrintTensorInfo(node_active);
    UpdateActive();
}
void HyperTreeBaseImpl::UpdateActive()
{
    // A culled node is not allowed to be active
    CHECK_EQ((this->node_active.cpu() * this->node_culled.cpu()).sum().item().toInt(), 0);

    auto node_active = this->node_active.to(torch::kCPU);
    std::vector<long> active_node_ids;
    std::vector<long> node_active_prefix_sum;
    int active_count = 0;
    for (int i = 0; i < NumNodes(); ++i)
    {
        if (node_active.data_ptr<int>()[i] == 1)
        {
            active_node_ids.push_back(i);
            node_active_prefix_sum.push_back(active_count);
            active_count++;
        }
        else
        {
            node_active_prefix_sum.push_back(-1);
        }
    }

    // Use set_data because otherwise the registered buffer is broken.
    this->active_node_ids.set_data(torch::from_blob(&active_node_ids[0], {(long)active_node_ids.size()},
                                                    torch::TensorOptions().dtype(torch::kLong))
                                       .clone()
                                       .to(this->device()));
    this->node_active_prefix_sum.set_data(torch::from_blob(&node_active_prefix_sum[0],
                                                           {(long)node_active_prefix_sum.size()},
                                                           torch::TensorOptions().dtype(torch::kLong))
                                              .clone()
                                              .to(this->device()));


    // this->active_node_ids        = this->active_node_ids.to(node_position_min.device());
    // this->node_active_prefix_sum = this->node_active_prefix_sum.to(node_position_min.device());
}

torch::Tensor HyperTreeBaseImpl::InactiveNodeIds()
{
    auto node_active = this->node_active.to(torch::kCPU);
    std::vector<long> inactive_node_ids;
    for (int i = 0; i < NumNodes(); ++i)
    {
        if (node_active.data_ptr<int>()[i] == 0)
        {
            inactive_node_ids.push_back(i);
        }
    }
    return torch::from_blob(&inactive_node_ids[0], {(long)inactive_node_ids.size()},
                            torch::TensorOptions().dtype(torch::kLong))
        .clone()
        .to(this->device());
}

void HyperTreeBaseImpl::SetErrorForActiveNodes(torch::Tensor error, std::string strategy)
{
    CHECK_EQ(error.sizes(), node_error.sizes());
    CHECK_EQ(error.device(), node_error.device());

    auto active_float   = node_active.to(torch::kFloat32);
    auto inactive       = (1 - active_float);
    auto invalid_errors = (node_error < 0).to(torch::kFloat32);
    // Set all elements except the active to 0
    // error = active_float * error * node_max_density;
    error = active_float * error;

    // std::cout << "Updating Tree error with strategy = " << strategy << std::endl;
    // PrintTensorInfo(node_error);

    if (strategy == "override")
    {
        // Set all active elements to 0
        auto new_node_error = node_error * inactive;
        new_node_error      = new_node_error + error;
        node_error.set_data(new_node_error);
    }
    else if (strategy == "min")
    {
        // Set the inactive elements of the new erros to a large value
        // so min() does not use them
        error += inactive * 1000000;

        // Set the active elements of the old data to a large value,
        // that are still initialized with -1
        auto filtered_error = node_error + (invalid_errors * active_float) * 1000000;

        auto new_node_error = torch::min(filtered_error, error);
        node_error.set_data(new_node_error);
    }
    else
    {
        CHECK(false);
    }
    // PrintTensorInfo(node_error);
}

std::vector<AABB> HyperTreeBaseImpl::ActiveNodeBoxes()
{
    CHECK(active_node_ids.is_cpu());
    std::vector<AABB> result(NumActiveNodes());

    auto active_node_ids_ptr = active_node_ids.data_ptr<long>();
    auto box_min_ptr         = node_position_min.data_ptr<vec3>();
    auto box_max_ptr         = node_position_max.data_ptr<vec3>();

    for (int i = 0; i < result.size(); ++i)
    {
        auto nid  = active_node_ids_ptr[i];
        result[i] = AABB(box_min_ptr[nid], box_max_ptr[nid]);
    }
    return result;
}

void HyperTreeBaseImpl::UpdateCulling()
{
    auto culled   = node_culled.cpu();
    auto children = node_children.cpu();
    auto active   = node_active.cpu();

    int* culled_ptr   = culled.data_ptr<int>();
    int* children_ptr = children.data_ptr<int>();
    int* active_ptr   = active.data_ptr<int>();

    bool changed = true;

    while (changed)
    {
        changed = false;

        for (int i = 0; i < culled.size(0); ++i)
        {
            if (culled_ptr[i]) continue;

            bool all_culled = true;
            for (int cid = 0; cid < NS(); ++cid)
            {
                int c = children_ptr[i * NS() + cid];
                if (!culled_ptr[c]) all_culled = false;
            }
            if (all_culled)
            {
                CHECK(!active_ptr[i]);
                culled_ptr[i] = true;
                changed       = true;
            }
        }
    }

    node_culled.set_data(culled.to(node_culled.device()));

    // Set error of all culled nodes to 0
    node_error.mul_(1 - node_culled);
}