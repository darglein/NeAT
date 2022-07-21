/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */
#include "cimg_wrapper.h"

#define cimg_display 0
#include "CImg.h"


torch::Tensor LoadHDRImageTensor(const std::string& path)
{
    using namespace cimg_library;
    CImg<float> image(path.c_str());

    int h = image.height();
    int w = image.width();
    int d = image.depth();

    torch::Tensor result = torch::zeros({1, d, h, w});
    float* data          = result.data_ptr<float>();

    for (int x = 0; x < w; ++x)
    {
        for (int y = 0; y < h; ++y)
        {
            for (int z = 0; z < d; ++z)
            {
                float f = image.data(x, y, z, 0)[0];

                data[z * (w * h) + y * w + x] = f;
            }
        }
    }

    return result;
}


void SaveHDRImageTensor(torch::Tensor volume, const std::string& path)
{
    using namespace cimg_library;
    // one channel volume
    CHECK_EQ(volume.dim(), 4);
    // CHECK_EQ(volume.size(0), 1);


    volume = volume.contiguous();


    CImg<float> image(volume.data_ptr<float>(), volume.size(3), volume.size(2), volume.size(1), volume.size(0));
    image.save_analyze(path.c_str());
}