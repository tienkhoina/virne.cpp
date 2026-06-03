#include "yaml_merge.h"

namespace yaml_merge {

YAML::Node merge(
    const YAML::Node& base,
    const YAML::Node& override)
{
    if (!base)
        return YAML::Clone(override);

    if (!override)
        return YAML::Clone(base);

    if (!base.IsMap() ||
        !override.IsMap())
    {
        return YAML::Clone(override);
    }

    YAML::Node result =
        YAML::Clone(base);

    for (auto it = override.begin();
         it != override.end();
         ++it)
    {
        auto key =
            it->first.as<std::string>();

        if (result[key] &&
            result[key].IsMap() &&
            it->second.IsMap())
        {
            result[key] =
                merge(
                    result[key],
                    it->second);
        }
        else
        {
            result[key] =
                YAML::Clone(
                    it->second);
        }
    }

    return result;
}

}