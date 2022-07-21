/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "saiga/cuda/cudaHelper.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "CameraPose.h"

#include <torch/autograd.h>

#include <torch/csrc/autograd/custom_function.h>

using namespace Saiga;
static __global__ void RotatePointGPUk(StaticDeviceTensor<double, 2> point, StaticDeviceTensor<int, 1> index,
                                       StaticDeviceTensor<double, 2> rotation, StaticDeviceTensor<double, 2> out_point)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point.sizes[0]) return;

    Vec3 position = ((Vec3*)&point(tid, 0))[0];
    int id        = index(tid);
    Quat q        = ((Quat*)&rotation(id, 0))[0];

    Vec3 new_pos = RotatePoint(q, position);

    ((Vec3*)&out_point(tid, 0))[0] = new_pos;
}


using namespace Saiga;
static __global__ void RotatePointBackwardGPUk(StaticDeviceTensor<double, 2> point, StaticDeviceTensor<int, 1> index,
                                               StaticDeviceTensor<double, 2> rotation,
                                               StaticDeviceTensor<double, 2> in_grad_point,
                                               StaticDeviceTensor<double, 2> out_grad_rotation_tangent,
                                               StaticDeviceTensor<double, 2> out_grad_point)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point.sizes[0]) return;

    Vec3 position         = ((Vec3*)&point(tid, 0))[0];
    Vec3 grad_in_position = ((Vec3*)&in_grad_point(tid, 0))[0];
    int id                = index(tid);
    Quat q                = ((Quat*)&rotation(id, 0))[0];

    Mat3 J_p, J_r;
    Vec3 new_pos = RotatePoint(q, position, &J_r, &J_p);


    Vec3 g_rot   = J_r.transpose() * grad_in_position;
    Vec3 g_point = J_p.transpose() * grad_in_position;

    for (int k = 0; k < 3; ++k)
    {
        atomicAdd(&out_grad_rotation_tangent(id, k), g_rot(k));
        atomicAdd(&out_grad_point(tid, k), g_point(k));
    }
}

namespace torch::autograd
{
struct RotatePoint : public Function<RotatePoint>
{
    // returns a tensor for every layer
    static variable_list forward(AutogradContext* ctx, Variable rotation, Variable rotation_tangent, Variable points,
                                 Variable index)
    {
        torch::Tensor result = torch::empty_like(points);
        RotatePointGPUk<<<iDivUp(points.size(0), 256), 256>>>(points, index, rotation, result);

        variable_list l;
        l.push_back(result);

        if (ctx)
        {
            std::vector<torch::Tensor> save_variables;
            save_variables.push_back(rotation);
            save_variables.push_back(points);
            save_variables.push_back(index);
            ctx->save_for_backward(save_variables);
        }

        CUDA_SYNC_CHECK_ERROR();
        return l;
        //        return {};
    }

    static variable_list backward(AutogradContext* ctx, variable_list input_gradients)
    {
        CHECK_EQ(input_gradients.size(), 1);
        auto in_grad_point = input_gradients[0];
        //        std::cout << "rotate backward " << input_gradients.size() << std::endl;
        //        for (auto t : input_gradients)
        //        {
        //            PrintTensorInfo(t);
        //        }

        auto saved_variables = ctx->get_saved_variables();
        CHECK_EQ(saved_variables.size(), 3);
        auto rotation = saved_variables[0];
        auto points   = saved_variables[1];
        auto index    = saved_variables[2];

        auto out_grad_rot_tangent =
            torch::zeros({rotation.size(0), 3}, torch::TensorOptions(rotation.device()).dtype(rotation.dtype()));

        auto out_grad_points =
            torch::zeros({points.size(0), 3}, torch::TensorOptions(points.device()).dtype(points.dtype()));


        RotatePointBackwardGPUk<<<iDivUp(points.size(0), 256), 256>>>(points, index, rotation, in_grad_point,
                                                                      out_grad_rot_tangent, out_grad_points);

        variable_list gradients;
        gradients.push_back({});
        gradients.push_back(out_grad_rot_tangent);
        gradients.push_back(out_grad_points);
        gradients.push_back({});
        CUDA_SYNC_CHECK_ERROR();
        return gradients;
    }
};
}  // namespace torch::autograd


namespace Saiga
{
CameraPoseModuleImpl::CameraPoseModuleImpl(Saiga::ArrayView<PoseType> poses)
{
    long N           = poses.size();
    rotation_tangent = torch::zeros({N, 3L}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    rotation_tangent = rotation_tangent.set_requires_grad(true);


    static_assert(sizeof(PoseType) == sizeof(double) * 8);
    static_assert(sizeof(Quat) == sizeof(double) * 4);
    static_assert(sizeof(Vec3) == sizeof(double) * 3);

    std::vector<Vec3> ts;
    std::vector<Quat> rs;

    for (auto p : poses)
    {
        ts.push_back(p.translation());
        rs.push_back(p.unit_quaternion());
    }


    translation = torch::from_blob(ts.data(), {N, 3L}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    rotation    = torch::from_blob(rs.data(), {N, 4L}, torch::TensorOptions().dtype(torch::kFloat64)).clone();

    register_buffer("rotation", rotation);
    register_parameter("rotation_tangent", rotation_tangent);
    register_parameter("translation", translation);
}


static __global__ void ApplyTangent(Saiga::Vec3* tangent, Sophus::SO3d* pose, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Saiga::Vec3 t = tangent[tid];
    auto p        = pose[tid];
    p             = Sophus::SO3<double>::exp(t) * p;

    pose[tid]    = p;
    tangent[tid] = Saiga::Vec3::Zero();
}

void CameraPoseModuleImpl::ApplyTangent()
{
    //   SAIGA_ASSERT(rotation_tangent.is_contiguous() && rotation.is_contiguous());
    int n = rotation.size(0);
    int c = Saiga::iDivUp(n, 128);
    Saiga::ApplyTangent<<<c, 128>>>((Saiga::Vec3*)rotation_tangent.data_ptr<double>(),
                                    (Sophus::SO3d*)rotation.data_ptr<double>(), n);
}



torch::Tensor CameraPoseModuleImpl::RotatePoint(torch::Tensor p, torch::Tensor index)
{
    CHECK_EQ(p.device(), rotation.device());


    auto list = torch::autograd::RotatePoint::apply(rotation, rotation_tangent, p, index);
    CHECK_EQ(list.size(), 1);
    return list.front();


    if (p.is_cuda())
    {
        //         return RotatePointGPU(p, index, rotation);
    }
    CHECK(rotation.is_cpu());
    torch::Tensor result = torch::empty_like(p);

    Vec3* res_ptr = result.data_ptr<Vec3>();
    Vec3* p_ptr   = p.data_ptr<Vec3>();
    int* iid_ptr  = index.data_ptr<int>();
    Quat* rot_ptr = rotation.data_ptr<Quat>();

    for (int i = 0; i < p.size(0); ++i)
    {
        Vec3 p     = p_ptr[i];
        Quat q     = rot_ptr[iid_ptr[i]];
        p          = q * p;
        res_ptr[i] = p;
    }

    return result;
}
void CameraPoseModuleImpl::AddNoise(double noise_translation, double noise_rotation)
{
    if (noise_translation > 0)
    {
        torch::NoGradGuard ngg;
        std::cout << "Adding Translational Noise" << std::endl;
        auto noise = (torch::rand_like(translation) - 0.5) * noise_translation;
        PrintTensorInfo(noise);
        translation.add_(noise);
    }
    if (noise_rotation > 0)
    {
        torch::NoGradGuard ngg;
        std::cout << "Adding Translational Noise" << std::endl;
        auto noise = (torch::rand_like(rotation_tangent) - 0.5) * noise_rotation;
        PrintTensorInfo(noise);
        rotation_tangent.add_(noise);
        ApplyTangent();
    }
}

CameraModelModuleImpl::CameraModelModuleImpl(int h, int w, ArrayView<IntrinsicsPinholed> data) : h(h), w(w)
{
    long N           = data.size();
    const long count = 5;
    using VectorType = Eigen::Matrix<double, count, 1>;
    std::vector<VectorType> intrinsics_data;

    for (auto& d : data)
    {
        VectorType coeffs = d.coeffs();
        intrinsics_data.push_back(coeffs);
    }

    intrinsics =
        torch::from_blob(intrinsics_data.data(), {N, count}, torch::TensorOptions().dtype(torch::kFloat64)).clone();


    register_parameter("intrinsics", intrinsics);
    std::cout << "Pinhole Intrinsics:" << std::endl;
    PrintTensorInfo(intrinsics);
    type = CameraModelType::PINHOLE;
}
torch::Tensor CameraModelModuleImpl::Unproject(torch::Tensor camera_index, torch::Tensor pixel_coords,
                                               torch::Tensor depth)
{
    depth        = depth.unsqueeze(1);
    auto cameras = torch::index_select(intrinsics, 0, camera_index);

    auto fxfy = cameras.slice(1, 0, 2);
    auto cxcy = cameras.slice(1, 2, 4);
    auto c    = (pixel_coords - cxcy) / fxfy;

    c = torch::cat({c, torch::ones_like(depth)}, 1);
    return c * depth;
}
std::vector<IntrinsicsPinholed> CameraModelModuleImpl::DownloadPinhole()
{
    long N           = intrinsics.size(0);
    const long count = 5;
    using VectorType = Eigen::Matrix<double, count, 1>;
    std::vector<IntrinsicsPinholed> intrinsics_data;

    auto intrinsics_cpu = intrinsics.cpu();
    auto intrinsics_ptr = intrinsics_cpu.data_ptr<double>();

    for (int i = 0; i < N; ++i)
    {
        VectorType v;
        for (int d = 0; d < count; ++d)
        {
            v(d) = intrinsics_ptr[i * count + d];
        }
        intrinsics_data.push_back(v);
    }

    return intrinsics_data;
}
}  // namespace Saiga