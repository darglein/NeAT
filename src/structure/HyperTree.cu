
/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */
 #define CUDA_NDEBUG

#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"

#include "HyperTree.h"


template <int D>
struct DeviceHyperTree
{
    using Vec = Eigen::Vector<float, D>;
    StaticDeviceTensor<float, 2> node_position_min;
    StaticDeviceTensor<float, 2> node_position_max;
    StaticDeviceTensor<long, 1> active_node_ids;
    StaticDeviceTensor<int, 1> node_active;
    StaticDeviceTensor<int, 2> node_children;

    __host__ DeviceHyperTree(HyperTreeBaseImpl* tree)
    {
        node_position_min = tree->node_position_min;
        node_position_max = tree->node_position_max;
        active_node_ids   = tree->active_node_ids;
        node_active       = tree->node_active;
        node_children     = tree->node_children;
        //        node_diagonal_length = tree->node_diagonal_length;
        CHECK_EQ(node_position_min.strides[1], 1);
        CHECK_EQ(node_position_max.strides[1], 1);
    }

    __device__ Vec PositionMin(int node_id)
    {
        Vec* ptr = (Vec*)&node_position_min(node_id, 0);
        return ptr[0];
    }
    __device__ Vec PositionMax(int node_id)
    {
        Vec* ptr = (Vec*)&node_position_max(node_id, 0);
        return ptr[0];
    }
};

template <int D>
static __global__ void ComputeLocalSamples(DeviceHyperTree<D> tree, StaticDeviceTensor<float, 3> global_samples,
                                           StaticDeviceTensor<long, 1> node_indices,
                                           StaticDeviceTensor<float, 3> out_local_samples)
{
    using Vec = typename DeviceHyperTree<D>::Vec;

    int group_id  = blockIdx.x;
    int local_tid = threadIdx.x;

    CUDA_KERNEL_ASSERT(group_id < global_samples.sizes[0]);
    int group_size = global_samples.sizes[1];
    int node_id    = node_indices(group_id);

    Vec pos_min = tree.PositionMin(node_id);
    Vec pos_max = tree.PositionMax(node_id);
    Vec size    = pos_max - pos_min;

    for (int i = local_tid; i < group_size; i += blockDim.x)
    {
        Vec c;
        for (int d = 0; d < D; ++d)
        {
            c(d) = global_samples(group_id, i, d);
        }
        c = c - pos_min;
        // [0, 1]
        c = c.array() / size.array();
        // [-1, 1]
        c = (c * 2) - Vec::Ones();

        ((Vec*)&out_local_samples(group_id, i, 0))[0] = c;
    }
}


template <int D>
static __global__ void ComputeLocalSamples2(DeviceHyperTree<D> tree, StaticDeviceTensor<float, 2> global_samples,
                                            StaticDeviceTensor<long, 1> node_indices,
                                            StaticDeviceTensor<float, 2> out_local_samples)
{
    using Vec = typename DeviceHyperTree<D>::Vec;

    int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample_id >= global_samples.sizes[0]) return;

    int node_id = node_indices(sample_id);

    Vec pos_min = tree.PositionMin(node_id);
    Vec pos_max = tree.PositionMax(node_id);
    Vec size    = pos_max - pos_min;


    Vec c;
    for (int d = 0; d < D; ++d)
    {
        c(d) = global_samples(sample_id, d);
    }
    c = c - pos_min;
    // [0, 1]
    c = c.array() / size.array();
    // [-1, 1]
    c = (c * 2) - Vec::Ones();

    ((Vec*)&out_local_samples(sample_id, 0))[0] = c;
}
torch::Tensor HyperTreeBaseImpl::ComputeLocalSamples(torch::Tensor global_samples, torch::Tensor node_indices)
{
    CHECK(global_samples.is_cuda());
    CHECK(node_position_min.is_cuda());


    auto local_samples = torch::zeros_like(global_samples);

    if (global_samples.dim() == 3 && node_indices.dim() == 1)
    {
#if 0
        CHECK_EQ(global_samples.dim(), 3);
        if (global_samples.size(0) > 0)
        {
            switch (D())
            {
                case 3:
                    ::ComputeLocalSamples<3>
                        <<<global_samples.size(0), 128>>>(this, global_samples, node_indices, local_samples);
                    break;
                default:
                    CHECK(false);
            }
        }
#endif


        auto sample_pos_min = torch::index_select(node_position_min, 0, node_indices).unsqueeze(1);
        auto sample_pos_max = torch::index_select(node_position_max, 0, node_indices).unsqueeze(1);
        auto size           = sample_pos_max - sample_pos_min;

        //    PrintTensorInfo(sample_pos_min);
        //    PrintTensorInfo(size);

        auto local_samples2 = (global_samples - sample_pos_min) / size * 2 - 1;

        //    PrintTensorInfo(local_samples);
        //    PrintTensorInfo(local_samples2);
        //    PrintTensorInfo(local_samples - local_samples2);

        CUDA_SYNC_CHECK_ERROR();
        return local_samples2;
    }
    else
    {
#if 0
        if (global_samples.size(0) > 0)
        {
            switch (D())
            {
                case 3:
                    ::ComputeLocalSamples2<3>
                        <<<global_samples.size(0), 128>>>(this, global_samples, node_indices, local_samples);
                    break;
                default:
                    CHECK(false);
            }
        }
#endif


        auto sample_pos_min = torch::index_select(node_position_min, 0, node_indices);
        auto sample_pos_max = torch::index_select(node_position_max, 0, node_indices);
        auto size           = sample_pos_max - sample_pos_min;

        // PrintTensorInfo(sample_pos_min);
        // PrintTensorInfo(size);

        auto local_samples2 = (global_samples - sample_pos_min) / size * 2 - 1;

        // PrintTensorInfo(local_samples);
        // PrintTensorInfo(local_samples2);
        // PrintTensorInfo(local_samples - local_samples2);
        // exit(0);

        CUDA_SYNC_CHECK_ERROR();
        return local_samples2;
    }

    // PrintTensorInfo(node_position_min);
    // PrintTensorInfo(node_position_max);
    // PrintTensorInfo(global_samples);
    // PrintTensorInfo(node_indices);
}

struct CompareInterval
{
    HD inline bool operator()(float kA, int vA, float kB, int vB) const { return kA < kB; }
};

template <int D, int ThreadsPerBlock>
static __global__ void ComputeRaySamples(
    DeviceHyperTree<D> tree, StaticDeviceTensor<float, 2> ray_origin, StaticDeviceTensor<float, 2> ray_direction,
    float* sample_rnd, int* out_num_samples,
    StaticDeviceTensor<float, 2> out_global_coordinates, StaticDeviceTensor<float, 1> out_weight,
    StaticDeviceTensor<float, 1> out_ray_t, StaticDeviceTensor<long, 1> out_ray_index,
    StaticDeviceTensor<long, 1> out_ray_local_id, StaticDeviceTensor<long, 1> out_node_id, int max_samples_per_node)
{
    using Vec     = typename DeviceHyperTree<D>::Vec;
    int ray_id    = blockIdx.x;
    int lane_id   = threadIdx.x % 32;
    int warp_id   = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    constexpr int max_intersections = 256;
    __shared__ int num_intersections;
    __shared__ int num_samples_of_ray;
    __shared__ int global_sample_offset;
    __shared__ int inter_num_samples[max_intersections];
    __shared__ int inter_num_samples_scan[max_intersections];
    __shared__ float inter_tmin[max_intersections];
    __shared__ float inter_tmax[max_intersections];
    __shared__ int inter_node_id[max_intersections];
    __shared__ int inter_id[max_intersections];

    if (threadIdx.x == 0)
    {
        num_intersections    = 0;
        num_samples_of_ray   = 0;
        global_sample_offset = 0;
    }

    for (int i = threadIdx.x; i < max_intersections; i += blockDim.x)
    {
        // Tmin is used for sorting therefore we need to set it far away
        inter_tmin[i]        = 12345678;
        inter_num_samples[i] = 0;
        inter_id[i]          = i;
    }

    __syncthreads();

    // Each block processes one ray
    Vec origin;
    Vec direction;
    for (int d = 0; d < D; ++d)
    {
        origin(d)    = ray_origin(ray_id, d);
        direction(d) = ray_direction(ray_id, d);
    }

    int num_active_nodes = tree.active_node_ids.sizes[0];

    // Use the complete block to test all active nodes
    for (int i = threadIdx.x; i < num_active_nodes; i += blockDim.x)
    {
        int node_id = tree.active_node_ids(i);

        Vec box_min, box_max;
        for (int d = 0; d < D; ++d)
        {
            box_min(d) = tree.node_position_min(node_id, d);
            box_max(d) = tree.node_position_max(node_id, d);
        }

        auto [hit, tmin, tmax] =
            IntersectBoxRayPrecise(box_min.data(), box_max.data(), origin.data(), direction.data(), D);

        if (tmax - tmin < 1e-5)
        {
            continue;
        }

        if (hit)
        {
            auto index = atomicAdd(&num_intersections, 1);
            CUDA_KERNEL_ASSERT(index < max_intersections);
            inter_tmin[index]    = tmin;
            inter_tmax[index]    = tmax;
            inter_node_id[index] = node_id;

            float diag               = (box_max - box_min).norm();
            float dis                = tmax - tmin;
            float rel_dis            = dis / diag;
            int num_samples          = iCeil(rel_dis * max_samples_per_node);
            inter_num_samples[index] = num_samples;

            atomicAdd(&num_samples_of_ray, num_samples);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        global_sample_offset = atomicAdd(out_num_samples, num_samples_of_ray);
        atomicMax(out_num_samples + 1, num_samples_of_ray);

        CUDA_KERNEL_ASSERT(global_sample_offset + num_samples_of_ray < out_weight.sizes[0]);
    }

    // TODO: more efficient scan
    for (int i = threadIdx.x; i < max_intersections; i += blockDim.x)
    {
        int count = 0;
        for (int j = 0; j < i; ++j)
        {
            int sorted_id   = inter_id[j];
            int num_samples = inter_num_samples[sorted_id];
            count += num_samples;
        }
        inter_num_samples_scan[i] = count;
    }

    __syncthreads();
    __syncthreads();

    // Process each intersection by a single warp
    for (int iid = warp_id; iid < num_intersections; iid += num_warps)
    {
        // the id and tmin was sorted therefore we can use iid
        int sorted_id        = inter_id[iid];
        float tmin           = inter_tmin[iid];
        int num_samples_scan = inter_num_samples_scan[iid];
        // these values have not been sorted -> use the sorted index to access them
        int node_id     = inter_node_id[sorted_id];
        float tmax      = inter_tmax[sorted_id];
        int num_samples = inter_num_samples[sorted_id];


        CUDA_KERNEL_ASSERT(num_samples <= max_samples_per_node);
        if (num_samples == 0) continue;

        float dis    = tmax - tmin;
        float weight = dis / num_samples;
        float step   = (tmax - tmin) / num_samples;


        int out_sample_index = 0;
        out_sample_index = global_sample_offset + num_samples_scan;

        for (int j = lane_id; j < num_samples; j += 32)
        {
            int global_sample_idx = out_sample_index + j;
            float t1              = tmin + j * step;
            float t2              = tmin + (j + 1) * step;

            float a = 0.5;
            if (sample_rnd)
            {
                a = sample_rnd[global_sample_idx];
            }
            float t        = t1 * (1 - a) + t2 * a;
            Vec global_pos = origin + t * direction;
            for (int d = 0; d < D; ++d)
            {
                out_global_coordinates(global_sample_idx, d) = global_pos(d);
            }
            out_ray_t(global_sample_idx)        = t;
            out_weight(global_sample_idx)       = weight;
            out_node_id(global_sample_idx)      = node_id;
            out_ray_index(global_sample_idx)    = ray_id;
            out_ray_local_id(global_sample_idx) = num_samples_scan + j;
        }

        __syncwarp();
        for (int j = lane_id; j < num_samples; j += 32)
        {
            float t = out_ray_t(out_sample_index + j);

            float t_half_min;
            if (j == 0)
                t_half_min = tmin;
            else
                t_half_min = (out_ray_t(out_sample_index + j - 1) + t) * 0.5f;

            float t_half_max;
            if (j == num_samples - 1)
                t_half_max = tmax;
            else
                t_half_max = (out_ray_t(out_sample_index + j + 1) + t) * 0.5f;

            out_weight(out_sample_index + j) = t_half_max - t_half_min;
        }
    }
}

SampleList HyperTreeBaseImpl::CreateSamplesForRays(const RayList& rays, int max_samples_per_node, bool interval_jitter)
{
    CHECK(rays.direction.is_cuda());
    CHECK(node_position_min.is_cuda());

    int predicted_samples = iCeil(rays.size() * max_samples_per_node * pow(NumActiveNodes(), 1.0 / D()));
    SampleList list;
    list.Allocate(predicted_samples, D(), node_position_min.device());


    torch::Tensor interval_rnd = interval_jitter ? torch::rand_like(list.weight) : torch::Tensor();
    float* interval_rnd_ptr    = interval_jitter ? interval_rnd.data_ptr<float>() : nullptr;

    auto out_num_samples_max_per_ray = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));

    switch (D())
    {
        case 3:
            ::ComputeRaySamples<3, 128><<<rays.size(), 128>>>(
                this, rays.origin, rays.direction, interval_rnd_ptr,
                out_num_samples_max_per_ray.data_ptr<int>(), list.global_coordinate, list.weight, list.ray_t,
                list.ray_index, list.local_index_in_ray, list.node_id, max_samples_per_node);
            break;
        default:
            CHECK(false);
    }

    out_num_samples_max_per_ray = out_num_samples_max_per_ray.cpu();
    int actual_samples          = out_num_samples_max_per_ray.data_ptr<int>()[0];
    list.max_samples_per_ray    = out_num_samples_max_per_ray.data_ptr<int>()[1];
    list.Shrink(actual_samples);
    CHECK_LE(actual_samples, predicted_samples);

    {
        // Use this method for position computation to get the correct positional gradient
        auto origin2           = torch::index_select(rays.origin, 0, list.ray_index);
        auto dir2              = torch::index_select(rays.direction, 0, list.ray_index);
        auto pos2              = origin2 + dir2 * list.ray_t.unsqueeze(1);
        list.global_coordinate = pos2;
    }

    // std::cout << "> CreateSamplesForRays: " << actual_samples << "/" << predicted_samples
    //           << " Max Per Ray: " << list.max_samples_per_ray << std::endl;
    CUDA_SYNC_CHECK_ERROR();
    return list;
}



static __global__ void CountSamplesPerNode(StaticDeviceTensor<long, 1> node_id,
                                           StaticDeviceTensor<int, 1> out_node_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_id.sizes[0]) return;
    int target_idx = node_id(tid);
    atomicAdd(&out_node_count(target_idx), 1);
}

static __global__ void ComputedPaddedCount(StaticDeviceTensor<int, 1> node_count,
                                           StaticDeviceTensor<int, 1> out_node_count_padded, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_count.sizes[0]) return;
    int count                  = node_count(tid);
    int padded_count           = iAlignUp(count, group_size);
    out_node_count_padded(tid) = padded_count;
}

static __global__ void ComputeIndexOrder(StaticDeviceTensor<long, 1> node_id,
                                         StaticDeviceTensor<long, 1> samples_per_node_scan,
                                         StaticDeviceTensor<int, 1> current_node_elements,
                                         StaticDeviceTensor<long, 1> per_group_node_id,
                                         StaticDeviceTensor<int, 1> src_indices,
                                         StaticDeviceTensor<float, 1> padding_weights, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_id.sizes[0]) return;
    int nid    = node_id(tid);
    int start  = nid == 0 ? 0 : samples_per_node_scan(nid - 1);
    int offset = atomicAdd(&current_node_elements(nid), 1);

    if (offset % group_size == 0)
    {
        // The first sample of each group writes the node id!
        int k = offset / group_size;

        per_group_node_id(start / group_size + k) = nid;
    }
    src_indices(start + offset)     = tid;
    padding_weights(start + offset) = 1;
}

NodeBatchedSamples HyperTreeBaseImpl::GroupSamplesPerNodeGPU(const SampleList& samples, int group_size)
{
    auto device = samples.global_coordinate.device();
    CHECK_EQ(device, active_node_ids.device());
    int num_samples = samples.size();
    int num_nodes   = NumNodes();
    // CHECK_GT(num_samples, 0);

    torch::Tensor num_samples_per_node = torch::zeros({num_nodes}, torch::TensorOptions(device).dtype(torch::kInt));

    if (num_samples > 0)
    {
        CountSamplesPerNode<<<iDivUp(num_samples, 256), 256>>>(samples.node_id, num_samples_per_node);
    }

    torch::Tensor num_samples_per_node_padded = torch::zeros_like(num_samples_per_node);
    ComputedPaddedCount<<<iDivUp(num_nodes, 256), 256>>>(num_samples_per_node, num_samples_per_node_padded, group_size);

    auto samples_per_node_scan = torch::cumsum(num_samples_per_node_padded, 0);
    int num_output_samples     = samples_per_node_scan.slice(0, num_nodes - 1, num_nodes).item().toLong();
    SAIGA_ASSERT(num_output_samples % group_size == 0);
    int num_groups = num_output_samples / group_size;

    torch::Tensor current_node_elements = torch::zeros({num_nodes}, torch::TensorOptions(device).dtype(torch::kInt));
    torch::Tensor per_group_node_id     = torch::zeros({num_groups}, torch::TensorOptions(device).dtype(torch::kLong));
    torch::Tensor src_indices = torch::zeros({num_output_samples}, torch::TensorOptions(device).dtype(torch::kInt));
    torch::Tensor padding_weights =
        torch::zeros({num_output_samples}, torch::TensorOptions(device).dtype(torch::kFloat));

    if (num_samples > 0)
    {
        ComputeIndexOrder<<<iDivUp(num_samples, 256), 256>>>(samples.node_id, samples_per_node_scan,
                                                             current_node_elements, per_group_node_id, src_indices,
                                                             padding_weights, group_size);
    }


    NodeBatchedSamples result;
    result.global_coordinate = torch::index_select(samples.global_coordinate, 0, src_indices);
    result.global_coordinate = result.global_coordinate.reshape({-1, group_size, D()});

    result.mask = padding_weights.reshape({-1, group_size, 1});

    result.integration_weight = torch::index_select(samples.weight, 0, src_indices) * padding_weights;
    result.integration_weight = result.integration_weight.reshape({-1, group_size, 1});

    result.node_ids = per_group_node_id;

    result.ray_index = torch::index_select(samples.ray_index, 0, src_indices);
    result.ray_index = result.ray_index.reshape({-1, group_size, 1});

    if (samples.local_index_in_ray.defined())
    {
        result.sample_index_in_ray = torch::index_select(samples.local_index_in_ray, 0, src_indices);
        result.sample_index_in_ray = result.sample_index_in_ray.reshape({-1, group_size, 1});
    }
    CUDA_SYNC_CHECK_ERROR();
    return result;
}


template <int D>
static __global__ void VolumeSamples(Eigen::Vector<int, D> size, StaticDeviceTensor<float, 2> out_position,
                                     StaticDeviceTensor<long, 1> out_index, bool swap_xy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_position.sizes[0]) return;

    Eigen::Vector<int, D> c;
    int tmp = tid;

    if (swap_xy)
    {
        for (int d = D - 1; d >= 0; --d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for (int d = 0; d < D; ++d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }

    Eigen::Vector<float, D> ones = Eigen::Vector<float, D>::Ones();
    Eigen::Vector<float, D> pos =
        (c.template cast<float>().array() / (size.template cast<float>() - ones).array()).eval();
    Eigen::Vector<float, D> global_coordinates = ((pos * 2).eval() - ones).eval();

    for (int d = 0; d < D; ++d)
    {
        out_position(tid, d) = global_coordinates(d);
    }
    out_index(tid) = tid;
}

SampleList HyperTreeBaseImpl::UniformPhantomSamplesGPU(Eigen::Vector<int, -1> size, bool swap_xy)
{
    int num_samples = size.array().prod();
    SampleList result;
    result.Allocate(num_samples, D(), device());

    VolumeSamples<3><<<iDivUp(num_samples, 256), 256>>>(size, result.global_coordinate, result.ray_index, swap_xy);


    std::tie(result.node_id, result.weight) = NodeIdForPositionGPU(result.global_coordinate);
    CUDA_SYNC_CHECK_ERROR();
    return result;
}

template <int D>
__device__ inline int ActiveNodeIdForGlobalPosition(DeviceHyperTree<D> tree, float* global_position)
{
    constexpr int NS = 1 << D;
    int current_node = 0;
    while (true)
    {
        int node_id = current_node;
        if (tree.node_active(current_node) > 0)
        {
            // Found an active node, use this to compute the local coordinates
            break;
        }

        // Not active -> decent into children which contains the sample
        for (int cid = 0; cid < NS; ++cid)
        {
            int c = tree.node_children(current_node, cid);

            if (c == -1)
            {
                // This sample is not inside an active node
                return -1;
            }
            CUDA_KERNEL_ASSERT(c >= 0);
            CUDA_KERNEL_ASSERT(c != node_id);
            float* pos_min = &tree.node_position_min(c, 0);
            float* pos_max = &tree.node_position_max(c, 0);

            if (BoxContainsPoint(pos_min, pos_max, global_position, D))
            {
                current_node = c;
                break;
            }
        }

        if (current_node == node_id)
        {
            return -1;
        }
        CUDA_KERNEL_ASSERT(current_node != node_id);
    }
    return current_node;
}

template <int D>
static __global__ void ComputeNodeId(StaticDeviceTensor<float, 2> global_samples, DeviceHyperTree<D> tree,
                                     StaticDeviceTensor<long, 1> out_index, StaticDeviceTensor<float, 1> out_mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= global_samples.sizes[0]) return;
    float* position  = &global_samples(tid, 0);
    int current_node = ActiveNodeIdForGlobalPosition(tree, position);

    if (current_node == -1)
    {
        // just use the first active node id
        out_index(tid) = tree.active_node_ids(0);
        out_mask(tid)  = 0;
    }
    else
    {
        out_index(tid) = current_node;
        out_mask(tid)  = 1;
    }
}
std::tuple<torch::Tensor, torch::Tensor> HyperTreeBaseImpl::NodeIdForPositionGPU(torch::Tensor global_samples)
{
    CHECK(global_samples.is_cuda());
    CHECK(node_position_min.is_cuda());
    auto samples_linear = global_samples.reshape({-1, D()});
    int num_samples     = samples_linear.size(0);

    auto result_node_id = torch::zeros({num_samples}, global_samples.options().dtype(torch::kLong));
    auto result_mask    = torch::zeros({num_samples}, global_samples.options().dtype(torch::kFloat));

    CHECK(samples_linear.is_contiguous());

    CHECK_EQ(D(), 3);
    if (num_samples > 0)
    {
        ComputeNodeId<3><<<iDivUp(num_samples, 256), 256>>>(samples_linear, this, result_node_id, result_mask);
    }
    CUDA_SYNC_CHECK_ERROR();
    return {result_node_id, result_mask};
}

template <int D>
static __global__ void InterpolateGridForInactiveNodes(DeviceHyperTree<D> tree,
                                                       StaticDeviceTensor<float, 5> active_grid,
                                                       StaticDeviceTensor<float, 5> out_full_grid)
{
    int node_id   = blockIdx.x;
    int thread_id = threadIdx.x;
    CUDA_KERNEL_ASSERT(node_id < active_grid.sizes[0]);

    using Vec = vec3;
    Vec box_min;
    Vec box_max;
    for (int d = 0; d < D; ++d)
    {
        box_min(d) = tree.node_position_min(node_id, d);
        box_max(d) = tree.node_position_max(node_id, d);
    }
    Vec box_size = box_max - box_min;

    if (!tree.node_active(node_id))
    {
        //        printf("processing active node %d\n", node_id);
        ivec3 grid_size(11, 11, 11);
        int total_cells = grid_size.array().prod();
        for (int cell_id = thread_id; cell_id < total_cells; cell_id += blockDim.x)
        {
            ivec3 res;
            int tmp = cell_id;
            for (int d = 0; d < D; ++d)
            {
                res(d) = tmp % grid_size(d);
                tmp /= grid_size(d);
            }

            // range [0,1]
            Vec local_coords = res.array().cast<float>() / (grid_size - ivec3::Ones()).array().cast<float>();


            // convert into global
            Vec c             = local_coords;
            c                 = c.array() * box_size.array();
            Vec global_coords = (c + box_min);

            int target_node = ActiveNodeIdForGlobalPosition(tree, global_coords.data());


            // convert into local space of target node
            Vec target_box_min;
            Vec target_box_max;
            for (int d = 0; d < D; ++d)
            {
                target_box_min(d) = tree.node_position_min(target_node, d);
                target_box_max(d) = tree.node_position_max(target_node, d);
            }
            Vec target_box_size = target_box_max - target_box_min;
            c                   = global_coords;
            c                   = c - target_box_min;
            c                   = c.array() / target_box_size.array();
            // range [0,1]
            Vec target_local_coords_01 = c;

            // compute nearest neighbor of target
            Vec feature_coord    = target_local_coords_01.array() * (grid_size - ivec3::Ones()).array().cast<float>();
            ivec3 feature_coordi = feature_coord.array().round().cast<int>();


            //            printf("nodes %d %d, %d %d %d -> %d %d %d\n", node_id, target_node, res(0), res(1), res(2),
            //                   feature_coordi(0), feature_coordi(1), feature_coordi(2));

            // copy into target
            for (int c = 0; c < out_full_grid.sizes[1]; ++c)
            {
                out_full_grid(node_id, c, res(2), res(1), res(0)) =
                    active_grid(target_node, c, feature_coordi(2), feature_coordi(1), feature_coordi(0));
            }
        }
    }
}

torch::Tensor HyperTreeBaseImpl::InterpolateGridForInactiveNodes(torch::Tensor active_grid)
{
    std::cout << "HyperTreeBaseImpl::InterpolateGridForInactiveNodes" << std::endl;

    // float [num_nodes, 8, 11, 11, 11]
    torch::Tensor interpolated = active_grid.clone();
    CHECK_EQ(D(), 3);
    CHECK_EQ(interpolated.size(2), 11);

    PrintTensorInfo(active_grid);
    ::InterpolateGridForInactiveNodes<3><<<NumNodes(), 128>>>(this, active_grid, interpolated);
    CUDA_SYNC_CHECK_ERROR();
    return interpolated;
}

template <int D>
static __global__ void UniformGlobalSamples(DeviceHyperTree<D> tree, StaticDeviceTensor<int, 1> node_ids,
                                            ivec3 grid_size, StaticDeviceTensor<float, 5> out_position)
{
    CUDA_KERNEL_ASSERT(blockIdx.x < node_ids.sizes[0]);

    int node_id   = node_ids((int)blockIdx.x);
    int thread_id = threadIdx.x;

    vec3 pos_min = tree.PositionMin(node_id);
    vec3 pos_max = tree.PositionMax(node_id);
    vec3 size    = pos_max - pos_min;

    int total_cells = grid_size.array().prod();
    for (int cell_id = thread_id; cell_id < total_cells; cell_id += blockDim.x)
    {
        ivec3 res;
        int tmp = cell_id;
        for (int d = 0; d < D; ++d)
        {
            res(d) = tmp % grid_size(d);
            tmp /= grid_size(d);
        }

        // in [0,1]
        vec3 local_position = res.cast<float>().array() / (grid_size - ivec3::Ones()).cast<float>().array();

        vec3 global_position = local_position.array() * size.array();
        global_position += pos_min;

        for (int d = 0; d < D; ++d)
        {
            out_position((int)blockIdx.x, res(2), res(1), res(0), d) = global_position(d);
        }
    }
}
torch::Tensor HyperTreeBaseImpl::UniformGlobalSamples(torch::Tensor node_id, int grid_size)
{
    CHECK_EQ(node_id.dim(), 1);

    int N   = node_id.size(0);
    node_id = node_id.to(torch::kInt);

    auto result_position = torch::zeros({N, grid_size, grid_size, grid_size, D()}, node_position_min.options());
    ::UniformGlobalSamples<3><<<N, 128>>>(this, node_id, ivec3(grid_size, grid_size, grid_size), result_position);

    return result_position;
}


template <int D>
static __global__ void NodeNeighborSamples(DeviceHyperTree<D> tree, Eigen::Vector<int, D> size, float epsilon,
                                           int* out_num_samples, StaticDeviceTensor<float, 2> out_global_coordinates,
                                           StaticDeviceTensor<float, 1> out_weight,
                                           StaticDeviceTensor<float, 1> out_ray_t,
                                           StaticDeviceTensor<long, 1> out_ray_index,
                                           StaticDeviceTensor<long, 1> out_ray_local_id,
                                           StaticDeviceTensor<long, 1> out_node_id)
{
    int node_id  = tree.active_node_ids((int)blockIdx.x);
    int local_id = threadIdx.x;


    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= out_position.sizes[0]) return;

    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 box_size = node_max - node_min;

    for (int side = 0; side < D; ++side)
    {
        int side_a = (side + 1) % 3;
        int side_b = (side + 2) % 3;

        for (int front = 0; front < 2; ++front)
        {
            vec3 add_eps(0, 0, 0);

            add_eps(side) = epsilon * (front ? 1 : -1);
            float rem     = (front ? 1 : 0);

            for (int sid = local_id; sid < size(side_a) * size(side_b); sid += blockDim.x)
            {
                // float x = 0;
                // float y = (sid % size(1)) / (size(1) - 1.f);
                // float z = (sid / size(1)) / (size(2) - 1.f);

                vec3 local;
                local(side)   = rem;
                local(side_a) = (sid % size(side_a)) / (size(side_a) - 1.f);
                local(side_b) = (sid / size(side_a)) / (size(side_b) - 1.f);



                vec3 global_coords        = (local.array() * box_size.array() + node_min.array());
                vec3 global_coords_offset = global_coords + add_eps;
                int other_id              = ActiveNodeIdForGlobalPosition(tree, global_coords_offset.data());

                if (other_id != -1 && other_id != node_id)
                {
                    // found pair
                    int out_index = atomicAdd(out_num_samples, 2);
                    CUDA_KERNEL_ASSERT(out_index <= out_global_coordinates.sizes[0]);

                    for (int d = 0; d < D; ++d)
                    {
                        out_global_coordinates(out_index + 0, d) = global_coords(d);
                        out_global_coordinates(out_index + 1, d) = global_coords(d);
                    }
                    out_ray_index(out_index + 0) = out_index + 0;
                    out_ray_index(out_index + 1) = out_index + 1;

                    out_ray_local_id(out_index + 0) = 0;
                    out_ray_local_id(out_index + 1) = 1;

                    out_node_id(out_index + 0) = node_id;
                    out_node_id(out_index + 1) = other_id;

                    out_weight(out_index + 0) = 1;
                    out_weight(out_index + 1) = 1;
                }

                // if (node_id == 0)
                // {
                //     printf("%f %f %f, %f %f %f, %d %d\n", local(0), local(1), local(2), add_eps(0), add_eps(1),
                //     add_eps(2),
                //            node_id, other_id);
                // }
            }
        }
    }
}

SampleList HyperTreeBaseImpl::NodeNeighborSamples(Eigen::Vector<int, -1> size, double epsilon)
{
    int predicted_samples = NumActiveNodes() * 6 * size(0) * size(0) * 2;

    SampleList list;
    list.Allocate(predicted_samples, D(), node_position_min.device());
    auto out_num_samples = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));

    switch (D())
    {
        case 3:
            ::NodeNeighborSamples<3><<<NumActiveNodes(), 128>>>(this, size, epsilon, out_num_samples.data_ptr<int>(),
                                                                list.global_coordinate, list.weight, list.ray_t,
                                                                list.ray_index, list.local_index_in_ray, list.node_id);
            break;
        default:
            CHECK(false);
    }

    out_num_samples    = out_num_samples.cpu();
    int actual_samples = out_num_samples.data_ptr<int>()[0];
    list.Shrink(actual_samples);
    CHECK_LE(actual_samples, predicted_samples);

    CUDA_SYNC_CHECK_ERROR();
    return list;
}
