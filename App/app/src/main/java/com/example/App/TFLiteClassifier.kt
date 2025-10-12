package com.example.App

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.sqrt

class TFLiteClassifier(
    private val absoluteModelPath: String,
    numThreads: Int = 2,
    useNNAPI: Boolean = false,
    private val norm: ModelUtils.NormType = ModelUtils.NormType.IMAGENET_STD
) {
    private val interpreter: Interpreter
    private var isNCHW: Boolean
    private var inputW: Int
    private var inputH: Int

    init {
        val modelFile = File(absoluteModelPath)
        if (!modelFile.exists()) {
            throw Exception("Model file does not exist at path: $absoluteModelPath")
        }

        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            if (useNNAPI) setUseNNAPI(true)
        }
        interpreter = Interpreter(modelFile, options)

        val inT = interpreter.getInputTensor(0)
        val inputShape = inT.shape()

        isNCHW = (inputShape.size == 4 && inputShape[1] == 3)
        inputH = if (isNCHW) inputShape[2] else inputShape[1]
        inputW = if (isNCHW) inputShape[3] else inputShape[2]
    }

    fun embed(bitmap: Bitmap): FloatArray {
        val input: ByteBuffer =
            if (isNCHW) ModelUtils.bitmapToNCHWFloat32Buffer(bitmap, inputW, inputH, norm)
            else ModelUtils.bitmapToNHWCFloat32Buffer(bitmap, inputW, inputH, norm)

        val out = TensorBuffer.createFixedSize(interpreter.getOutputTensor(0).shape(), DataType.FLOAT32)
        interpreter.run(input, out.buffer)

        val v = out.floatArray
        val normValue = sqrt(v.fold(0.0f) { acc, x -> acc + x * x })
        if (normValue > 0f) for (i in v.indices) v[i] /= normValue
        return v
    }

    fun close() = interpreter.close()
}