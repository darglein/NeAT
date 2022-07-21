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


torch::Tensor LoadHDRImageTensor(const std::string& path);

void SaveHDRImageTensor(torch::Tensor volume, const std::string& path);