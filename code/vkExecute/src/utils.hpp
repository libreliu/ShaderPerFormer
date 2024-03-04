#pragma once

#include <vector>
#include <algorithm>
#include <iostream>

namespace vkExecute{
    int editDistance(const std::vector<int>& arr1, const std::vector<int>& arr2) {
        if (arr1.size() < arr2.size())
            return editDistance(arr2, arr1);

        std::vector<int> prev(arr2.size() + 1, 0);
        std::vector<int> curr(arr2.size() + 1, 0);

        for (int j = 0; j <= arr2.size(); ++j)
            prev[j] = j;

        for (int i = 1; i <= arr1.size(); ++i) {
            curr[0] = i;
            for (int j = 1; j <= arr2.size(); ++j) {
                if (arr1[i - 1] == arr2[j - 1])
                    curr[j] = prev[j - 1];
                else
                    curr[j] = 1 + std::min({ prev[j], curr[j - 1], prev[j - 1] });
            }
            std::swap(prev, curr);
        }

        return prev[arr2.size()];
    }
}

