package com.example.App

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ModelUtils {
    enum class DataLayout { NCHW, NHWC }
    enum class NormType { NEG_ONE_TO_ONE, ZERO_TO_ONE, IMAGENET_STD }

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun bitmapToFloat32Buffer(
        bitmap: Bitmap, width: Int, height: Int, layout: DataLayout,
        norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer = when (layout) {
        DataLayout.NCHW -> bitmapToNCHWFloat32Buffer(bitmap, width, height, norm)
        DataLayout.NHWC -> bitmapToNHWCFloat32Buffer(bitmap, width, height, norm)
    }

    @JvmOverloads
    fun bitmapToNCHWFloat32Buffer(
        bitmap: Bitmap, width: Int, height: Int, norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * 3 * height * width).order(ByteOrder.nativeOrder())
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val intValues = IntArray(width * height)
        resized.getPixels(intValues, 0, width, 0, 0, width, height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                buffer.putFloat(normR((p shr 16) and 0xFF, norm))
            }
        }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                buffer.putFloat(normG((p shr 8) and 0xFF, norm))
            }
        }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = intValues[y * width + x]
                buffer.putFloat(normB(p and 0xFF, norm))
            }
        }
        return buffer.rewind() as ByteBuffer
    }

    @JvmOverloads
    fun bitmapToNHWCFloat32Buffer(
        bitmap: Bitmap, width: Int, height: Int, norm: NormType = NormType.NEG_ONE_TO_ONE
    ): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * height * width * 3).order(ByteOrder.nativeOrder())
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val intValues = IntArray(width * height)
        resized.getPixels(intValues, 0, width, 0, 0, width, height)
        for (p in intValues) {
            buffer.putFloat(normR((p shr 16) and 0xFF, norm))
            buffer.putFloat(normG((p shr 8) and 0xFF, norm))
            buffer.putFloat(normB(p and 0xFF, norm))
        }
        return buffer.rewind() as ByteBuffer
    }

    private fun normR(v: Int, norm: NormType): Float = when (norm) {
        NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
        NormType.ZERO_TO_ONE    -> v / 255f
        NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    }
    private fun normG(v: Int, norm: NormType): Float = when (norm) {
        NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
        NormType.ZERO_TO_ONE    -> v / 255f
        NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    }
    private fun normB(v: Int, norm: NormType): Float = when (norm) {
        NormType.NEG_ONE_TO_ONE -> (v - 127.5f) / 127.5f
        NormType.ZERO_TO_ONE    -> v / 255f
        NormType.IMAGENET_STD   -> ((v / 255f) - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    }
}