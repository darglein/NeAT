/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/
#pragma once

#include "SceneBase.h"
#include "Settings.h"
#include "structure/HyperTree.h"

using namespace Saiga;


struct SampleData
{
    int scene_id = 0;
    PixelList pixels;
    int NumPixels() { return pixels.uv.size(0); }
};


struct RowSampleData
{
    PixelList pixels;

    int batch_size = 0;

    // only for eval
    int image_id  = 0;
    int row_start = 0;
    int row_end   = 0;

    int NumPixels() { return pixels.uv.size(0); }
};

class RandomMultiSceneDataset : public torch::data::BatchDataset<RandomMultiSceneDataset, SampleData>
{
   public:
    RandomMultiSceneDataset(std::vector<std::vector<int>> indices, std::vector<std::shared_ptr<SceneBase>> scene,
                            std::shared_ptr<CombinedParams> params, int rays_per_image)
        : indices_list(indices), scene_list(scene), params(params), rays_per_image(rays_per_image)
    {
        CHECK_GT(rays_per_image, 0);
        total_num_images = 0;
        for (auto& i : indices)
        {
            total_num_images += i.size();
        }
    }
    virtual torch::optional<size_t> size() const override { return total_num_images * rays_per_image; }

    virtual SampleData get_batch(torch::ArrayRef<size_t> indices);

   private:
    std::vector<std::vector<int>> indices_list;
    std::vector<std::shared_ptr<SceneBase>> scene_list;
    std::shared_ptr<CombinedParams> params;
    int rays_per_image;
    int total_num_images;
};


class RowRaybasedSampleDataset : public torch::data::BatchDataset<RowRaybasedSampleDataset, RowSampleData>
{
   public:
    RowRaybasedSampleDataset(std::vector<int> indices, std::shared_ptr<SceneBase> scene,
                             std::shared_ptr<CombinedParams> params, int out_w, int out_h, int rows_per_batch)
        : indices(indices), scene(scene), params(params), out_w(out_w), out_h(out_h), rows_per_batch(rows_per_batch)
    {
    }

    virtual torch::optional<size_t> size() const override
    {
        // Return one row at a time
        int num_batches_per_image = iDivUp(out_h, rows_per_batch);
        return indices.size() * num_batches_per_image;
    }

    virtual RowSampleData get_batch(torch::ArrayRef<size_t> indices);

   private:
    std::vector<int> indices;
    std::shared_ptr<SceneBase> scene;
    std::shared_ptr<CombinedParams> params;
    int out_w, out_h;
    int rows_per_batch;
};
