/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <map>

namespace MindStudioDebugger {

enum class MindStudioExtensionArgs {
    ALL_KERNEL_NAMES = 0,         /* const std::vector<std::string> --> char** */
    IS_KBK = 1,                   /* bool */

    /* Add before this line */
    ARG_MAX,
};

using ExtArgs = std::map<MindStudioExtensionArgs, void*>;

template <typename T>
T GetExtArgs(ExtArgs& args, MindStudioExtensionArgs id)
{
    auto it = args.find(id);
    if (it == args.end()) {
        return nullptr;
    }

    return reinterpret_cast<T>(it->second);
}

}