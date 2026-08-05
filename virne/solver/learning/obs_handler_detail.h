#pragma once

#include "obs_handler.h"

namespace virne::solver::learning::rl_core::detail {

using AttributeRegistryId = network::attribute::AttributeRegistryId;
using AttributeBenchmarkMap = network::attribute::AttributeBenchmarkMap;
using LinkAggregation = network::attribute::LinkAggregation;

void require_workers(std::size_t workers);

std::vector<AttributeRegistryId> select_node_ids(
    const network::BaseNetwork& network,
    const ObservationHandler::AttributeKinds& kinds);

std::vector<AttributeRegistryId> select_link_ids(
    const network::BaseNetwork& network,
    const ObservationHandler::AttributeKinds& kinds);

std::vector<double> resolve_node_benchmarks(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const AttributeBenchmarkMap& benchmarks);

std::vector<double> resolve_link_benchmarks(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const AttributeBenchmarkMap& benchmarks);

DenseMatrixF32 node_attrs_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<double>* benchmarks,
    std::size_t workers);

DenseMatrixF32 link_attrs_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<double>* benchmarks,
    std::size_t workers);

DenseMatrixF32 link_aggr_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    LinkAggregation aggregation,
    const std::vector<double>* benchmarks,
    std::size_t workers);

DenseMatrixF32 concatenate_columns(
    std::size_t rows,
    const std::vector<const DenseMatrixF32*>& matrices);

DenseVectorF32 concatenate_vectors(
    const std::vector<const DenseVectorF32*>& vectors);

DenseMatrixF32 empty_matrix(std::size_t rows);
DenseMatrixF32 constant_column(std::size_t rows, float value);

std::vector<AttrId> resolve_node_value_ids(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids);

std::vector<AttrId> resolve_link_value_ids(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids);

DenseVectorF32 v_node_demand_by_id(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<AttrId>& value_ids,
    const std::vector<double>* benchmarks);

DenseVectorF32 v_node_aggr_by_id(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    LinkAggregation aggregation,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<AttrId>& value_ids,
    const std::vector<double>* benchmarks);

}  // namespace virne::solver::learning::rl_core::detail
