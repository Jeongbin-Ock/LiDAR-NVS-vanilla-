/*
 * Copyright (C) 2025, Fudan Zhang Vision Group
 * All rights reserved.
 * 
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 4 // None, None, intensity, raydrop
// Per-Gaussian feature channels (features.size(1)) one launch may blend.
// The rendered normal adds three more output channels after them.
#define MAX_FEATURE_CHANNELS 32
// Feature counts compiled as constants (accumulators in registers): none,
// and V18's 30. Any other count up to MAX_FEATURE_CHANNELS takes the
// run-time path, which keeps them in local memory.
#define SPECIALIZED_FEATURE_CHANNELS 30
#define BLOCK_X 16
#define BLOCK_Y 16

#endif