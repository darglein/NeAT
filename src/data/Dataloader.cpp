/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/
#include "Dataloader.h"

SampleData RandomMultiSceneDataset::get_batch(torch::ArrayRef<size_t> sample_indices)
{
    int scene_id  = Random::uniformInt(0, indices_list.size() - 1);
    auto& indices = indices_list[scene_id];
    auto& scene   = scene_list[scene_id];


    std::vector<PixelList> pixels;
    torch::Tensor rays_per_image = torch::zeros({(long)indices.size()}, torch::kLong);
    torch::set_num_threads(1);

    int total_num_rays          = sample_indices.size();
    torch::Tensor ray_image_ids = torch::randint(0, indices.size(), {total_num_rays}).to(torch::kLong);
    rays_per_image.scatter_add_(0, ray_image_ids, torch::ones_like(ray_image_ids));

    for (int i = 0; i < rays_per_image.size(0); ++i)
    {
        int image_id  = indices[i];
        int ray_count = rays_per_image.data_ptr<long>()[i];
        if (ray_count == 0) continue;
        std::shared_ptr<UnifiedImage> img = scene->frames[image_id];
        auto& cam                         = scene->cameras[img->camera_id];
        CHECK(img->projection.defined());

        PixelList result;

            result.uv = img->CoordinatesRandomNoInterpolate(ray_count, cam.w, cam.h);

        std::tie(result.target, result.target_mask) = img->SampleProjection(result.uv);
        result.image_id                             = torch::full({result.uv.size(0)}, (int)image_id,
                                                                  torch::TensorOptions(torch::kInt32).device(result.uv.device()));
        result.camera_id                            = torch::full({result.uv.size(0)}, img->camera_id,
                                                                  torch::TensorOptions(torch::kInt32).device(result.uv.device()));
        pixels.push_back(result);
    }

    SampleData sample_data;
    sample_data.scene_id = scene_id;
    sample_data.pixels   = PixelList(pixels);
    sample_data.pixels.to(device);
    return sample_data;
}

RowSampleData RowRaybasedSampleDataset::get_batch(torch::ArrayRef<size_t> sample_indices)
{
    torch::set_num_threads(1);

    std::vector<int> sample_image_ids;

    int row_start = 0;
    int row_end   = 0;

    CHECK_GT(indices.size(), 0);

    int num_batches_per_image = iDivUp(out_h, rows_per_batch);


    CHECK_EQ(sample_indices.size(), 1);
    int image_id = sample_indices.front() / num_batches_per_image;
    int batch_id = sample_indices.front() % num_batches_per_image;
    row_start    = batch_id * rows_per_batch;
    row_end      = std::min(out_h, row_start + rows_per_batch);
    image_id     = indices[image_id];
    sample_image_ids.push_back(image_id);


    std::vector<PixelList> pixels;
    for (auto image_id : sample_image_ids)
    {
        std::shared_ptr<UnifiedImage> img = scene->frames[image_id];
        CHECK(img->projection.defined());

        PixelList result;
        result.uv                                   = img->CoordinatesRow(row_start, row_end, out_w, out_h);
        std::tie(result.target, result.target_mask) = img->SampleProjection(result.uv);
        result.image_id                             = torch::full({result.uv.size(0)}, (int)image_id,
                                                                  torch::TensorOptions(torch::kInt32).device(result.uv.device()));
        result.camera_id                            = torch::full({result.uv.size(0)}, img->camera_id,
                                                                  torch::TensorOptions(torch::kInt32).device(result.uv.device()));
        pixels.push_back(result);
    }


    RowSampleData sample_data;
    sample_data.batch_size = sample_image_ids.size();
    sample_data.row_start  = row_start;
    sample_data.row_end    = row_end;
    sample_data.image_id   = sample_image_ids.front();
    sample_data.pixels     = PixelList(pixels);
    sample_data.pixels.to(device);
    return sample_data;
}
