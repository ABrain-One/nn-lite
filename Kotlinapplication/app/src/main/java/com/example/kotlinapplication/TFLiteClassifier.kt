package com.example.kotlinapplication

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import kotlin.math.sqrt

class TFLiteClassifier(
    context: Context,
    private val modelPath: String = "dpn107_model_0.1204.tflite",
    numThreads: Int = 1,
    useNNAPI: Boolean = false,
    private val norm: ModelUtils.NormType = ModelUtils.NormType.IMAGENET_STD
) {
    private val interpreter: Interpreter
    private var inputShape: IntArray
    private var outputShape: IntArray
    private var isNCHW: Boolean
    private var inputW: Int
    private var inputH: Int

    init {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setUseXNNPACK(false); setUseNNAPI(false); setNumThreads(numThreads)
            if (useNNAPI) runCatching { setUseNNAPI(true) }
        }
        interpreter = Interpreter(model, options)

        var inT = interpreter.getInputTensor(0)
        var outT = interpreter.getOutputTensor(0)
        inputShape = inT.shape(); outputShape = outT.shape()

        isNCHW = (inputShape.size == 4 && (inputShape[1] == 3 || inputShape[1] == 1))
        val c = if (isNCHW) inputShape[1] else inputShape[3]
        val h = if (isNCHW) inputShape[2] else inputShape[1]
        val w = if (isNCHW) inputShape[3] else inputShape[2]
        if (inputShape[0] != 1) {
            val newShape = if (isNCHW) intArrayOf(1, c, h, w) else intArrayOf(1, h, w, c)
            Log.w("Embed", "Resizing input ${inputShape.contentToString()} -> ${newShape.contentToString()}")
            interpreter.resizeInput(0, newShape)
            interpreter.allocateTensors()
            inT = interpreter.getInputTensor(0); outT = interpreter.getOutputTensor(0)
            inputShape = inT.shape(); outputShape = outT.shape()
        }

        if (isNCHW) { inputH = inputShape[2]; inputW = inputShape[3] }
        else        { inputH = inputShape[1]; inputW = inputShape[2] }

        Log.d("Embed", "FINAL input=${inputShape.contentToString()} output=${outputShape.contentToString()}")
    }

    /** Returns L2-normalized embedding (FloatArray of length D). */
    fun embed(bitmap: Bitmap): FloatArray {
        val input: ByteBuffer =
            if (isNCHW) ModelUtils.bitmapToNCHWFloat32Buffer(bitmap, inputW, inputH, norm)
            else         ModelUtils.bitmapToNHWCFloat32Buffer(bitmap, inputW, inputH, norm)

        val out = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        out.buffer.rewind()
        interpreter.run(input, out.buffer)

        val v = out.floatArray
        val norm = sqrt(v.fold(0.0) { acc, x -> acc + (x * x) }.toFloat())
        if (norm > 0f) for (i in v.indices) v[i] /= norm
        return v
    }

    fun close() = runCatching { interpreter.close() }
}
