package com.example.kotlinapplication

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ModelUtils {

    /** Data layout for model input tensor. */
    enum class DataLayout { NCHW, NHWC }

    /** Normalization to apply to input pixels. */
    enum class NormType {
        /** (x - 127.5) / 127.5  -> [-1, 1] */
        NEG_ONE_TO_ONE,
        /** x / 255 -> [0, 1] */
        ZERO_TO_ONE,
        /** (x/255 - mean)/std with ImageNet per-channel stats */
        IMAGENET_STD
    }

    // ImageNet means/stds on 0..1 scale
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f) // R, G, B
    private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Generic entry: convert Bitmap -> Float32 ByteBuffer for either layout with chosen normalization.
     *
     * @param bitmap Input image (any size; will be resized)
     * @param width  Target width expected by the model
     * @param height Target height expected by the model
     * @param layout Data layout of model input tensor (NCHW or NHWC)
     * @param norm   Normalization type (default [-1,1])
     */
    fun bitmapToFloat32Buffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        layout: DataLayout,
        norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer {
        return when (layout) {
            DataLayout.NCHW -> bitmapToNCHWFloat32Buffer(bitmap, width, height, norm)
            DataLayout.NHWC -> bitmapToNHWCFloat32Buffer(bitmap, width, height, norm)
        }
    }

    /**
     * Convert a Bitmap into a Float32 ByteBuffer in NCHW format (1, 3, H, W).
     * Default normalization: [-1, 1].
     */
    @JvmOverloads
    fun bitmapToNCHWFloat32Buffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * 1 * 3 * height * width).order(ByteOrder.nativeOrder())

        // Resize
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)

        // Extract pixels
        val intValues = IntArray(width * height)
        resized.getPixels(intValues, 0, width, 0, 0, width, height)

        // Helpers for normalization
        fun normR(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        }
        fun normG(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        }
        fun normB(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }

        // Channel-first planes: R then G then B
        // R
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                val r = (p shr 16) and 0xFF
                buffer.putFloat(normR(r))
            }
        }
        // G
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                val g = (p shr 8) and 0xFF
                buffer.putFloat(normG(g))
            }
        }
        // B
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                val b = p and 0xFF
                buffer.putFloat(normB(b))
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Convert a Bitmap into a Float32 ByteBuffer in NHWC format (1, H, W, 3).
     * Default normalization: [-1, 1].
     */
    @JvmOverloads
    fun bitmapToNHWCFloat32Buffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * 1 * height * width * 3).order(ByteOrder.nativeOrder())

        // Resize
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)

        // Extract pixels
        val intValues = IntArray(width * height)
        resized.getPixels(intValues, 0, width, 0, 0, width, height)

        fun normR(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        }
        fun normG(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        }
        fun normB(v: Int): Float = when (norm) {
            NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
            NormType.ZERO_TO_ONE    -> v / 255f
            NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }

        // Pixel-wise interleaved: R, G, B
        var idx = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[idx++]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                buffer.putFloat(normR(r))
                buffer.putFloat(normG(g))
                buffer.putFloat(normB(b))
            }
        }

        buffer.rewind()
        return buffer
    }
}
