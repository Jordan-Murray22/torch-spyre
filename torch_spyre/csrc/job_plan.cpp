/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "job_plan.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "flex/src/runtime_stream/operations/runtime_operation_host_produce.hpp"
#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "spyrecode-host-functions/processSpyreCodeArtifacts.h"

namespace spyre {

void JobPlanStepH2D::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  auto* params =
      flex::createDmaParams(host_address_, device_address_.total_size(),
                            /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device CompositeAddress: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    const auto& device_address =
        std::get<flex::CompositeAddress>(device_address_);
    auto* params =
        flex::createDmaParams(host_address_, device_address.total_size(),
                              /*to_device=*/false, &device_address);
    params->pipeline_barrier = pipeline_barrier_;
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  } else {
    const uint64_t dmva = std::get<Dmva>(device_address_).value;
    auto segment_id = flex::dmvaToSegmentId(dmva);
    TORCH_CHECK(segment_id < ctx.inputs_outputs.size(),
                "D2H tensor-segment lookup out of range: segment ", segment_id,
                " but only ", ctx.inputs_outputs.size(),
                " launch args were provided");
    const auto& tensor = ctx.inputs_outputs.at(segment_id);
    const auto& tensor_address =
        static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
            ->composite_addr;
    TORCH_CHECK(tensor_address.chunks().size() == 1,
                "Tensor address must have 1 chunk");
    const auto& base_chunk = tensor_address.chunks()[0];
    uint64_t segment_offset = dmva - (segment_id << flex::SEGMENT_SIZE_BITS);
    TORCH_CHECK(segment_offset + size_ <= tensor_address.total_size(),
                "D2H transfer out of bounds: offset ", segment_offset,
                " + size ", size_, " exceeds tensor allocation size ",
                tensor_address.total_size());
    flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                     base_chunk.addr.offset + segment_offset);
    flex::Chunk offset_chunk(offset_addr, size_, base_chunk.domain_id);

    // Create shared_ptr to manage lifetime - will be kept alive by callback
    auto device_address =
        std::make_shared<flex::CompositeAddress>(offset_chunk);

    auto* params =
        flex::createDmaParams(host_address_, device_address->total_size(),
                              /*to_device=*/false, device_address.get());
    params->pipeline_barrier = pipeline_barrier_;
    params->callback = [device_address](void*) {};
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  }
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    os << "    Device CompositeAddress: "
       << std::get<flex::CompositeAddress>(device_address_) << "\n";
  } else {
    os << "    Device dmva: " << std::get<Dmva>(device_address_).value << "\n";
  }
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      tensor_allocs.push_back(address);
    }
  }
  auto* params = flex::createComputeParams(
      &program_address_, std::move(tensor_allocs), name_, bootstrap_offset_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchCompute(params);
  flex::destroyComputeParams(params);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program CompositeAddress: " << program_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::vector<int64_t> JobPlanStepHostCompute::resolveSymbolicArgs(
    const std::vector<at::Tensor>& tensors,
    const std::vector<SymbolicArg>& symbolic_args) {
  auto& allocator = SpyreAllocator::instance();
  std::vector<int64_t> resolved(symbolic_args.size());
  for (size_t i = 0; i < symbolic_args.size(); ++i) {
    const SymbolicArg& arg = symbolic_args[i];
    TORCH_CHECK(arg.tensor_id >= 0 &&
                    static_cast<size_t>(arg.tensor_id) < tensors.size(),
                "SymbolicArg[", i, "].tensor_id=", arg.tensor_id,
                " out of range [0, ", tensors.size(), ")");
    switch (arg.kind) {
      case SymbolicArgKind::kAddress:
        resolved[i] = static_cast<int64_t>(allocator.compositeAddressToDmva(
            static_cast<SharedOwnerCtx*>(
                tensors[arg.tensor_id].storage().data_ptr().get_context())
                ->composite_addr));
        break;
      case SymbolicArgKind::kDimension:
        TORCH_CHECK(false,
                    "SymbolicArgKind::kDimension is not yet implemented");
        break;
      default:
        TORCH_CHECK(false, "Unknown SymbolicArgKind value: ",
                    static_cast<int32_t>(arg.kind));
    }
  }
  return resolved;
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  // Alignment for the staged RaiiBuffer (matches RaiiBuffer's own page-size
  // default; will be upgraded to device IOVA alignment inside
  // handleDmaDataConversionH2D when a real device handle is present).
  static constexpr size_t kAlign = 4096;

  // Build the producer body.  All three source cases produce the same type
  // (RaiiBuffer) via different fill strategies; the kind label is always
  // "correction" for logs/profiler.
  flex::RuntimeOperationHostProduce::Producer producer;

  if (input_buffer_ != nullptr) {
    // Case 1: input_buffer_ is provided — use it directly as the source.
    producer = [this]() -> std::shared_ptr<flex::RaiiBuffer> {
      auto buf = std::make_shared<flex::RaiiBuffer>(correction_size_, kAlign);
      deeptools::processComputeOnHostCommand(*hcm_, buf->Pointer(),
                                             input_buffer_);
      return buf;
    };
  } else if (ishape_.size() == 1 && ishape_[0] == 0) {
    // Case 2: fake symbols (ishape_ is {0}) — nullptr src argument.
    producer = [this]() -> std::shared_ptr<flex::RaiiBuffer> {
      auto buf = std::make_shared<flex::RaiiBuffer>(correction_size_, kAlign);
      deeptools::processComputeOnHostCommand(*hcm_, buf->Pointer(), nullptr);
      return buf;
    };
  } else if (!ctx.symbolic_args.empty()) {
    // Case 3a: typed symbolic payload — resolve addresses by kind.
    std::vector<int64_t> resolved_addresses =
        resolveSymbolicArgs(ctx.inputs_outputs, ctx.symbolic_args);

    // Wrong symbolic_args count is an OOB read inside deeptools
    // (DT_CHECK_MSG_OPT is compiled out by default).
    TORCH_CHECK(resolved_addresses.size() == hcm_->vdci.inputSym_.size(),
                "symbolic_args count (", resolved_addresses.size(),
                ") does not match compiled symbol count (",
                hcm_->vdci.inputSym_.size(), ") for this host-compute step");

    producer = [this,
                resolved_addresses]() -> std::shared_ptr<flex::RaiiBuffer> {
      auto buf = std::make_shared<flex::RaiiBuffer>(correction_size_, kAlign);
      deeptools::processComputeOnHostCommand(*hcm_, buf->Pointer(),
                                             &resolved_addresses);
      return buf;
    };
  } else {
    // Case 3b: no payload — legacy path: treat every context tensor as an
    // address source in iteration order.  Back-compat for callers that pass no
    // symbolic_args (empty payload).
    std::vector<int64_t> addresses(ctx.inputs_outputs.size());
    int addr_idx = 0;
    auto& allocator = SpyreAllocator::instance();
    for (auto& tensor : ctx.inputs_outputs) {
      int64_t addr = static_cast<int64_t>(allocator.compositeAddressToDmva(
          (static_cast<SharedOwnerCtx*>(
               tensor.storage().data_ptr().get_context())
               ->composite_addr)));
      addresses[addr_idx++] = addr;
    }

    producer = [this, addresses]() -> std::shared_ptr<flex::RaiiBuffer> {
      auto buf = std::make_shared<flex::RaiiBuffer>(correction_size_, kAlign);
      // Use fast path with all tensor addresses.
      deeptools::processComputeOnHostCommandFast(
          fast_plan_, *hcm_, buf->Pointer(), addresses.data(),
          addresses.size());
      return buf;
    };
  }

  // Produce the staged buffer on the caller thread (same thread that produced
  // it before via the inline HostCallback synchronize path).
  flex::RuntimeOperationHostProduce produce_op(std::move(producer),
                                               /*kind=*/"correction");
  auto staged = produce_op.produce();

  // Launch the correction H2D with the staged buffer attached as the source.
  // The RaiiBuffer lifetime is extended by the completion callback set below.
  auto* params = flex::createDmaParams(staged->Pointer(), correction_size_,
                                       /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  // Keep staged alive until the DMA engine finishes reading it.
  params->callback = [staged](void*) { /* staged freed here */ };
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Correction size: " << correction_size_ << " bytes\n";
  os << "    Device address: " << device_address_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Fast path: "
     << (fast_plan_.valid
             ? "enabled"
             : (fast_plan_.output_size == UINT32_MAX ? "disabled" : "building"))
     << "\n";
  if (fast_plan_.valid) {
    os << "    Fast plan: " << fast_plan_.patches.size() << " patches, "
       << fast_plan_.num_input_symbols << " input symbols, "
       << fast_plan_.output_size << " bytes output\n";
  }
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation: " << addr << "\n";
    } else {
      os << "Program " << addr_idx - 1 << ": " << addr << "\n";
    }
    ++addr_idx;
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre
