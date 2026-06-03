#include <iostream>

#include "config/config_loader.h"
#include "config/override_parser.h"

static void print_direct(
    const Config& cfg,
    const char* stage)
{
    std::cout
        << "\n===== "
        << stage
        << " =====\n";

    try
    {
        std::cout
            << "solver.solver_name = "
            << cfg.get<std::string>(
                   "solver.solver_name")
            << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout
            << "solver ERROR: "
            << e.what()
            << '\n';
    }

    try
    {
        std::cout
            << "p_net_setting.topology.num_nodes = "
            << cfg.get<int>(
                   "p_net_setting.topology.num_nodes")
            << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout
            << "p_net ERROR: "
            << e.what()
            << '\n';
    }

    try
    {
        std::cout
            << "experiment.seed = "
            << cfg.get<int>(
                   "experiment.seed")
            << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout
            << "experiment ERROR: "
            << e.what()
            << '\n';
    }
}

int main(int argc, char** argv)
{
    try
    {
        Config cfg =
            ConfigLoader::load(
                "setting/main.yaml");

        print_direct(
            cfg,
            "BEFORE APPLY");

        cfg.save("before.yaml");
        

        for (int i = 1; i < argc; ++i)
        {
            std::cout
                << "\napply: "
                << argv[i]
                << '\n';

            override_parser::apply(
                cfg,
                argv[i]);
        }

        print_direct(
            cfg,
            "AFTER APPLY");

        cfg.save("after.yaml");

        std::cout
            << "\n===== ROOT DUMP =====\n";

        std::cout
            << cfg.root()
            << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\nFATAL:\n"
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}