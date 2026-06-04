#include "connectivity.h"

#include <boost/graph/connected_components.hpp>

namespace nx
{

bool is_connected(
    const Graph& g)
{
    if (g.num_nodes() == 0)
    {
        return true;
    }

    std::vector<int> component(
        g.num_nodes());

    int num_components =
        boost::connected_components(
            g.raw(),
            component.data());

    return num_components == 1;
}

}