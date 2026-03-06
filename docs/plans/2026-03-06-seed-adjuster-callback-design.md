# Seed Adjuster Callback for deduceMMASchedule

**Goal:** Replace `useGfx950Tuning` flag on `GPUMMAHeuristicSeeds` with a
`llvm::function_ref` callback, keeping target-specific logic out of the
generic GPU heuristics layer and enabling future per-architecture adjusters.

## Type Alias

In `GPUHeuristics.h`:

```cpp
using SeedAdjustFn = llvm::function_ref<void(
    const GPUMatmulShapeType &problem, const GPUIntrinsicType &intrinsic,
    std::optional<int64_t> wgpCount, GPUMMAHeuristicSeeds &seeds,
    int64_t splitReductionTripCnt)>;
```

Pass `GPUMMAHeuristicSeeds &seeds` (not individual fields) so future
adjusters can touch any seed without a signature change.

## deduceMMASchedule Signature

```cpp
FailureOr<GPUMMASchedule> deduceMMASchedule(
    ..., int64_t splitReductionTripCnt = 0,
    SeedAdjustFn seedAdjuster = adjustSeedsForWgpCount);
```

Default is `adjustSeedsForWgpCount` (the base adjuster). No null check
needed — always callable.

## Call Site in deduceMMASchedule

```cpp
GPUMMAHeuristicSeeds localSeeds = seeds;
seedAdjuster(problem, intrinsic, wgpCount, localSeeds,
             splitReductionTripCnt);
```

No branching. The callback replaces the previous if/else dispatch.

## File Changes

- **GPUHeuristics.h**: Add `SeedAdjustFn` type alias. Declare
  `adjustSeedsForWgpCount` (public, default adjuster). Remove
  `useGfx950Tuning` from `GPUMMAHeuristicSeeds`. Add `SeedAdjustFn`
  param to `deduceMMASchedule`.

- **GPUHeuristics.cpp**: Make `adjustSeedsForWgpCount` non-static. Adapt
  its signature to match `SeedAdjustFn` (take `GPUMMAHeuristicSeeds &`).
  Remove `adjustGfx950SeedsForWgpCount`. Simplify `deduceMMASchedule` call
  site to unconditionally call `seedAdjuster`.

- **ConfigUtils.cpp**: Define `adjustGfx950SeedsForWgpCount` locally
  (static). Adapt its signature to match `SeedAdjustFn`. When `isCDNA4`,
  pass it to `deduceMMASchedule`; otherwise use default. Remove
  `seeds.useGfx950Tuning` assignment.
