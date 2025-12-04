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
    numThreads: Int = 2,
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

    init {
        val modelFile = File(absoluteModelPath)
        if (!modelFile.exists()) {
            throw Exception("Model file does not exist at path: $absoluteModelPath")
        }

        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            when (delegateType) {
                DelegateType.CPU -> {} // Default
                DelegateType.GPU -> {
                    // Use default GPU delegate options to avoid compatibility issues
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                }
                DelegateType.NNAPI -> {
                    nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate)
                }
            }
        }
        interpreter = Interpreter(modelFile, options)

        val inT = interpreter.getInputTensor(0)
        var inputShape = inT.shape()

        // Force batch size to 1 if it's not
        if (inputShape[0] != 1) {
            val newShape = inputShape.clone()
            newShape[0] = 1
            interpreter.resizeInput(0, newShape)
            interpreter.allocateTensors()
            inputShape = newShape
            android.util.Log.d("TFLiteClassifier", "Resized input to: ${inputShape.contentToString()}")
        }

        // Handle batch dimension
        batchSize = inputShape[0]
        
        isNCHW = (inputShape.size == 4 && inputShape[1] == 3)
        inputH = if (isNCHW) inputShape[2] else inputShape[1]
        inputW = if (isNCHW) inputShape[3] else inputShape[2]
        
        android.util.Log.d("TFLiteClassifier", "Model Input Shape: ${inputShape.contentToString()}, BatchSize: $batchSize, W: $inputW, H: $inputH")
    }

    fun embed(bitmap: Bitmap): FloatArray {
        // Generate buffer for one image
        val input: ByteBuffer =
            if (isNCHW) ModelUtils.bitmapToNCHWFloat32Buffer(bitmap, inputW, inputH, norm)
            else ModelUtils.bitmapToNHWCFloat32Buffer(bitmap, inputW, inputH, norm)

        val out = TensorBuffer.createFixedSize(interpreter.getOutputTensor(0).shape(), DataType.FLOAT32)
        interpreter.run(input, out.buffer)

        val v = out.floatArray
        // Normalize only the first result if batch > 1? Or average? 
        // For benchmarking, we just want to run inference. 
        // But the return value is used. Let's just return the first vector.
        // If output is [Batch, Features], we take the first row.
        
        // Assuming output is flat or [Batch, Features]
        // We just take the whole array and normalize it? 
        // The original code calculated norm on the whole array. 
        // Let's keep it simple for now, as we care about duration.
        
        val normValue = sqrt(v.fold(0.0f) { acc, x -> acc + x * x })
        if (normValue > 0f) for (i in v.indices) v[i] /= normValue
        return v
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
    }
}