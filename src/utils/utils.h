/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "Settings.h"

#include "build_config.h"
#include "tensorboard_logger.h"

using namespace Saiga;

inline std::string EncodeImageToString(const Image& img, std::string type = "png")
{
    auto data = img.saveToMemory(type);

    std::string result;
    result.resize(data.size());

    memcpy(result.data(), data.data(), data.size());
    return result;
}

template <typename T>
inline void LogImage(TensorBoardLogger* tblogger, const TemplatedImage<T>& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}

inline void LogImage(TensorBoardLogger* tblogger, const Image& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}
