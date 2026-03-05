// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_VECTORIZEIREELINALGEXTOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeStaticMapStoreOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    if (mapStoreOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store is already vectorized");
    }
    ShapedType inputType = mapStoreOp.getInputType();
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store has non-static shape");
    }
    const int64_t innerSize = inputType.getShape()[inputType.getRank() - 1];
    const int64_t bitWidth = inputType.getElementTypeBitWidth();
    if ((innerSize * bitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store on sub-byte type");
    }
    // In case of a sub-byte bitwidth, we check that there is a contiguous copy
    // on the inner dimension that is a multiple of a byte. Note that the mask
    // shouldn't depend on the inner index for this.
    if (bitWidth < 8) {
      // First check that the mask is not the forward slice of the inner index.
      Value innermostInputIdx =
          mapStoreOp.getInputIndex(mapStoreOp.getInputRank() - 1);
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      Operation *maskOp = mapStoreOp.getMask().getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          mapStoreOp.getOutputIndex(mapStoreOp.getOutputRank() - 1);
      if (!isUnitFunctionOf(innermostOutputIdx, innermostInputIdx)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
    }
    Location loc = mapStoreOp.getLoc();
    rewriter.setInsertionPoint(mapStoreOp);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = vector::TransferReadOp::create(
        rewriter, loc, inputVectorType, mapStoreOp.getInput(),
        /*indices=*/zeros,
        /*padding=*/std::nullopt);
    auto vectorizedMapStoreOp =
        clone(rewriter, mapStoreOp, mapStoreOp.getResultTypes(),
              {inputVector, mapStoreOp.getOutput()});
    rewriter.replaceOp(mapStoreOp, vectorizedMapStoreOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Im2col Vectorization Helpers
//===----------------------------------------------------------------------===//

/// Compute the padding mask and adjusted read indices for im2col
/// vectorization. Thin wrapper around computeIm2colPaddingBounds: gets
/// clamped read offsets and valid size, then converts to a vector mask.
///
/// With non-wrapping delinearization in computeIm2colSourceIndices,
/// out-of-bounds output positions (from GEMM alignment padding) produce
/// out-of-bounds source coordinates that computeIm2colPaddingBounds
/// naturally handles via its spatial and channel dim bounds checks.
static Value computeIm2colPaddingMask(
    OpBuilder &b, Location loc, Im2colOp im2colOp,
    const Im2colSourceIndices &srcIndices, ArrayRef<OpFoldResult> inputSizes,
    ArrayRef<OpFoldResult> padLow, int64_t vecWidth,
    ArrayRef<Value> outputIVs, ArrayRef<OpFoldResult> outputOffsets,
    std::optional<int64_t> vecOutputDim, SmallVector<Value> &readIndices) {
  int64_t inputRank = im2colOp.getInputRank();
  auto vecI1Type = VectorType::get({vecWidth}, b.getI1Type());

  OpFoldResult innerTileSize = b.getIndexAttr(vecWidth);
  Im2colPaddingBounds bounds = computeIm2colPaddingBounds(
      b, loc, im2colOp, srcIndices, inputSizes, padLow, innerTileSize,
      outputIVs, outputOffsets, vecOutputDim);

  for (int64_t d = 0; d < inputRank; ++d) {
    readIndices.push_back(
        getValueOrCreateConstantIndexOp(b, loc, bounds.readOffsets[d]));
  }

  return vector::CreateMaskOp::create(b, loc, vecI1Type, bounds.validSize);
}

/// Vectorization pattern for im2col ops.
///
/// Matches a tiled im2col op with static output shape and directly emits
/// vector.transfer_read + vector.transfer_write ops, bypassing the
/// decompose-then-vectorize flow (loops + extract_slice + linalg.copy +
/// insert_slice + vectorize copy).
///
/// The pattern:
/// 1. Chooses a vectorization dimension (most contiguous in input)
/// 2. Iterates over all non-vectorized output positions
/// 3. For each position, computes source indices and emits vector ops
///
/// When padding is present (hasPadding() == true):
/// - Computes effective input sizes (unpadded + padding) for basis computation
/// - Adjusts source coordinates for reading from the unpadded input
/// - Emits masked vector.transfer_read with combined source + result mask
///
/// Falls back to scalar unrolling (vec_width=1) when no dimension is
/// vectorizable.
struct VectorizeIm2colOpPattern final
    : OpRewritePattern<IREE::LinalgExt::Im2colOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    // Require static output shape for vectorization.
    ShapedType outputType = im2colOp.getOutputType();
    if (!outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "im2col has non-static output shape");
    }

    Location loc = im2colOp.getLoc();
    bool hasPadding = im2colOp.hasPadding();

    // Get offsets from unified attributes (used for vectorization choice).
    SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();

    // Compute input sizes.
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());

    // Set up padding: zero vectors when no padding, actual values when present.
    int64_t inputRank = im2colOp.getInputRank();
    SmallVector<OpFoldResult> padLow(inputRank, rewriter.getIndexAttr(0));
    if (hasPadding) {
      SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
      if (!inputPadLow.empty()) {
        padLow = inputPadLow;
      }
    }

    // Choose vectorization dimension (uses unpadded sizes for contiguity).
    SmallVector<Range> iterationDomain(
        im2colOp.getIterationDomain(rewriter));
    std::optional<int64_t> maybeVecDim = chooseDimToVectorize(
        rewriter, loc, im2colOp, iterationDomain, inputSizes, mixedOffsets);

    int64_t outputRank = im2colOp.getOutputRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = outputType.getElementType();

    // Determine vectorization width and dimension.
    std::optional<int64_t> vecDim = maybeVecDim;
    int64_t vecWidth = vecDim ? outputShape[*vecDim] : 1;

    auto vecType = VectorType::get({vecWidth}, elemType);

    // Check that the vectorized input dimension has no padding. The source
    // padding mask uses scalar bounds checks which cannot correctly mask
    // individual vector lanes straddling a padding boundary.
    if (hasPadding && vecWidth > 1) {
      int64_t vecInputDim = inputRank - 1;
      SmallVector<OpFoldResult> inputPadHigh = im2colOp.getMixedInputPadHigh();
      OpFoldResult padHighVecDim =
          inputPadHigh.empty() ? rewriter.getIndexAttr(0)
                               : inputPadHigh[vecInputDim];
      if (!isConstantIntValue(padLow[vecInputDim], 0) ||
          !isConstantIntValue(padHighVecDim, 0)) {
        return rewriter.notifyMatchFailure(
            im2colOp,
            "padding on the vectorized input dimension is not supported");
      }
    }

    // Determine pad value for vector.transfer_read.
    Value padValue;
    if (hasPadding) {
      padValue = im2colOp.getPadValue();
    } else {
      padValue = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    }

    // Build permutation map for vector.transfer_write.
    int64_t writeDim = vecDim ? *vecDim : (outputRank - 1);
    AffineMap writePermMap = AffineMap::get(
        outputRank, 0, rewriter.getAffineDimExpr(writeDim),
        rewriter.getContext());

    // Enumerate all positions in non-vectorized dimensions.
    SmallVector<int64_t> loopDims;
    SmallVector<int64_t> loopBounds;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (vecDim && d == *vecDim) {
        continue;
      }
      loopDims.push_back(d);
      loopBounds.push_back(outputShape[d]);
    }

    int64_t totalIters = 1;
    for (int64_t bound : loopBounds) {
      totalIters *= bound;
    }

    // Maximum number of iterations that the vectorizer will statically unroll.
    // Exceeding this falls back to non-vectorized decomposition.
    static constexpr int64_t kMaxVectorizeUnrollIters = 1024;
    if (totalIters > kMaxVectorizeUnrollIters) {
      return rewriter.notifyMatchFailure(
          im2colOp, "im2col output too large for static unrolling");
    }

    Value result = im2colOp.getOutput();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    for (int64_t iter = 0; iter < totalIters; ++iter) {
      // Delinearize the flat iteration index into per-dimension indices.
      SmallVector<Value> ivs(outputRank, zeroIdx);
      int64_t remaining = iter;
      for (int64_t i = loopDims.size() - 1; i >= 0; --i) {
        int64_t idx = remaining % loopBounds[i];
        remaining /= loopBounds[i];
        ivs[loopDims[i]] =
            arith::ConstantIndexOp::create(rewriter, loc, idx);
      }

      // Map output position to input source coordinates.
      Im2colSourceIndices srcIndices = computeIm2colSourceIndices(
          rewriter, loc, im2colOp, ivs, rewriter.getIndexAttr(vecWidth));

      // Compute read indices and optional mask for the vector.transfer_read.
      SmallVector<Value> readIndices;
      Value mask;
      if (hasPadding) {
        // Padded: compute clamped read offsets and combined padding mask.
        // Output padding is only possible when hasPadding() is true (the
        // FoldOutputPadIntoIm2col pattern preserves the pad value).
        ArrayRef<Value> outputIVs;
        ArrayRef<OpFoldResult> outputOffsets;
        std::optional<int64_t> vecOutDim;
        if (hasOutputPadding(im2colOp)) {
          outputIVs = ivs;
          outputOffsets = mixedOffsets;
          if (vecDim) {
            vecOutDim = *vecDim;
          }
        }
        mask = computeIm2colPaddingMask(
            rewriter, loc, im2colOp, srcIndices, inputSizes, padLow,
            vecWidth, outputIVs, outputOffsets, vecOutDim, readIndices);
      } else {
        // Non-padded fast path: use source indices directly with no mask.
        for (OpFoldResult ofr : srcIndices.sliceOffsets) {
          readIndices.push_back(
              getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
        }
      }

      // Read a vector from the input tensor at the computed position.
      Value readVec;
      if (mask) {
        AffineMap readPermMap = AffineMap::getMinorIdentityMap(
            inputRank, 1, rewriter.getContext());
        auto inBoundsAttr = rewriter.getBoolArrayAttr({true});
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            readPermMap, padValue, mask, inBoundsAttr);
      } else {
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            padValue);
      }

      // Write the vector into the output tensor at the current position.
      SmallVector<Value> writeIndices(ivs);
      if (vecDim) {
        writeIndices[*vecDim] = zeroIdx;
      }
      result = vector::TransferWriteOp::create(rewriter, loc, readVec, result,
                                               writeIndices, writePermMap)
                   .getResult();
    }

    rewriter.replaceOp(im2colOp, result);
    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapStoreOpPattern>(context);
    patterns.add<VectorizeIm2colOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
