#pragma once

#include "../utils/setting.h"

namespace virne::network::detail {

virne::utils::SettingDocument clone_setting_document(
    const virne::utils::SettingDocument& source);

}  // namespace virne::network::detail
