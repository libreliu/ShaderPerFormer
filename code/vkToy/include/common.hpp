#pragma once

#define VKTOY_WARN(X, ...) fprintf(stderr, "vkToy: WARN: " X "\n", ##__VA_ARGS__)
#define VKTOY_LOG(X, ...) printf("vkToy: LOG: " X "\n", ##__VA_ARGS__)
