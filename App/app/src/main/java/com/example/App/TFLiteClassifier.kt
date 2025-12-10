package com.example.App

import android.graphics.Bitmap
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.sqrt

class TFLiteClassifier(
    private val absoluteModelPath: String,
    private val delegateType: DelegateType = DelegateType.CPU,
    numThreads: Int = 4,
    private val norm: ModelUtils.NormType = ModelUtils.NormType.IMAGENET_STD
) {
    enum class DelegateType { CPU, GPU, NNAPI }

    private val interpreter: Interpreter
    private var isNCHW: Boolean
    private var inputW: Int
    private var inputH: Int
    private var batchSize: Int
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var actualDelegateUsed: DelegateType

    init {
        val modelFile = File(absoluteModelPath)
        if (!modelFile.exists()) {
            throw Exception("Model file does not exist at path: $absoluteModelPath")
        }

        actualDelegateUsed = delegateType

        // Create interpreter based on delegate type
        interpreter = when (delegateType) {
            DelegateType.CPU -> {
                android.util.Log.d("TFLiteClassifier", "CPU: Using $numThreads threads with XNNPACK")
                val options = Interpreter.Options().apply {
                    setNumThreads(numThreads)
                    setUseXNNPACK(true)
                }
                Interpreter(modelFile, options)
            }
            DelegateType.GPU -> {
                // GPU with FP16 - no fallback, let errors propagate
                android.util.Log.d("TFLiteClassifier", "GPU: Attempting initialization...")
                val gpuOptions = GpuDelegate.Options().apply {
                    setPrecisionLossAllowed(true)
                }
                gpuDelegate = GpuDelegate(gpuOptions)
                val options = Interpreter.Options().addDelegate(gpuDelegate)
                val gpuInterpreter = Interpreter(modelFile, options)
                android.util.Log.d("TFLiteClassifier", "GPU: Initialized successfully")
                gpuInterpreter
            }
            DelegateType.NNAPI -> {
                // NNAPI - no fallback, let errors propagate
                android.util.Log.d("TFLiteClassifier", "NNAPI: Attempting initialization...")
                nnApiDelegate = NnApiDelegate()
                val options = Interpreter.Options().addDelegate(nnApiDelegate)
                val nnapiInterpreter = Interpreter(modelFile, options)
                android.util.Log.d("TFLiteClassifier", "NNAPI: Initialized successfully")
                nnapiInterpreter
            }
        }

        val inT = interpreter.getInputTensor(0)
        var inputShape = inT.shape()

        if (inputShape[0] != 1) {
            val newShape = inputShape.clone()
            newShape[0] = 1
            interpreter.resizeInput(0, newShape)
            interpreter.allocateTensors()
            inputShape = newShape
        }

        batchSize = inputShape[0]
        isNCHW = (inputShape.size == 4 && inputShape[1] == 3)
        inputH = if (isNCHW) inputShape[2] else inputShape[1]
        inputW = if (isNCHW) inputShape[3] else inputShape[2]
        
        android.util.Log.d("TFLiteClassifier", "Model ready: ${inputShape.contentToString()}, Delegate: $actualDelegateUsed")
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

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
    }

    fun getActualDelegate(): DelegateType = actualDelegateUsed
}