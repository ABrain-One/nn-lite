package com.example.App

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.ProgressBar
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import android.os.PowerManager
import android.content.Context

class MainActivity : AppCompatActivity() {

    private val ANDROID_STUDIO_VERSION = "Hedgehog | 2023.1.1" // CHANGE THIS MANUALLY
    private val TAG = "TFLiteRunner"
    private val DEVICE_MODEL_DIRECTORY = "/data/local/tmp"
    @DrawableRes private val DUMMY_IMAGE_RES_ID = R.drawable.car

    // UI Elements
    private lateinit var modelNameText: TextView
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements from the new layout
        modelNameText = findViewById(R.id.modelNameText)
        statusText = findViewById(R.id.statusText)
        progressBar = findViewById(R.id.progressBar)

        // Acquire wake lock to prevent app from being killed during long benchmarks
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        val wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "TFLiteBenchmark::BenchmarkWakeLock"
        )
        wakeLock.acquire(10 * 60 * 1000L /* 10 minutes */)

        runAutomaticBenchmark()
    }

    private fun runAutomaticBenchmark() {
        lifecycleScope.launch(Dispatchers.IO) {
            val modelFileName = intent.getStringExtra("model_filename")
            val reportJson = JSONObject()

            // Pre-populate JSON
            reportJson.put("model_name", modelFileName?.removeSuffix(".tflite") ?: "UNKNOWN")
            reportJson.put("device_type", Build.MODEL)
            reportJson.put("os_version", ANDROID_STUDIO_VERSION)
            reportJson.put("valid", true) // Always reports success
            reportJson.put("emulator", Build.FINGERPRINT.startsWith("generic") || Build.MODEL.contains("sdk") || Build.MODEL.contains("emulator"))

            withContext(Dispatchers.Main) {
                modelNameText.text = modelFileName?.removeSuffix(".tflite") ?: "Unknown Model"
                statusText.text = "Benchmark in progress..."
                progressBar.isVisible = true
            }

            if (modelFileName.isNullOrEmpty()) {
                Log.e(TAG, "FATAL: App launched without a 'model_filename' argument.")
                return@launch
            }

            val modelDevicePath = "$DEVICE_MODEL_DIRECTORY/$modelFileName"
            val inputBitmap = decodeResBitmapSafe(DUMMY_IMAGE_RES_ID, 512)
                ?: throw Exception("Failed to decode the dummy input image.")

            val delegates = listOf(
                "CPU" to TFLiteClassifier.DelegateType.CPU,
                "GPU" to TFLiteClassifier.DelegateType.GPU,
                "NPU" to TFLiteClassifier.DelegateType.NNAPI
            )

            val results = JSONObject()

            // Verify model file exists ONCE before running any benchmarks
            val modelFile = File(modelDevicePath)
            if (!modelFile.exists()) {
                Log.e(TAG, "❌ Model file does not exist at path: $modelDevicePath")
                val errorObj = JSONObject()
                errorObj.put("error", "Model file not found on device")
                results.put("CPU", errorObj)
                results.put("GPU", errorObj)
                results.put("NPU", errorObj)
                reportJson.put("results", results)
                reportJson.put("duration", -1)
                reportJson.put("valid", false)
                saveJsonReport(modelFileName, reportJson)
                
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Model file not found on device"
                    progressBar.isVisible = false
                }
                return@launch
            }
            Log.d(TAG, "✓ Model file verified: ${modelFile.absolutePath} (${modelFile.length()} bytes)")

            for ((name, type) in delegates) {
                try {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Running $name benchmark..."
                    }

                    // Timeout: 90s for CPU, 60s for GPU/NPU
                    val timeoutMs = if (type == TFLiteClassifier.DelegateType.CPU) 90000L else 60000L
                    
                    // Run benchmark with timeout using CompletableDeferred
                    val result = CompletableDeferred<Long?>()
                    
                    // Launch blocking work in a separate coroutine
                    val job = launch(Dispatchers.Default) {
                        try {
                            val classifier = TFLiteClassifier(modelDevicePath, delegateType = type)
                            classifier.embed(inputBitmap) // Warmup
                            val startTime = System.nanoTime()
                            classifier.embed(inputBitmap)
                            val totalNs = System.nanoTime() - startTime
                            classifier.close()
                            Log.d(TAG, "SUCCESS: $name Benchmark completed for $modelFileName. Total: ${totalNs}ns")
                            result.complete(totalNs)
                        } catch (e: Exception) {
                            Log.e(TAG, "$name benchmark error: ${e.message}")
                            result.complete(null)
                        }
                    }
                    
                    // Launch timeout watcher
                    val timeoutJob = launch {
                        delay(timeoutMs)
                        if (!result.isCompleted) {
                            Log.w(TAG, "⚠️ $name benchmark timed out after ${timeoutMs/1000}s - skipping")
                            result.complete(null)
                            job.cancel()
                        }
                    }
                    
                    // Wait for result
                    val benchmarkResult = result.await()
                    timeoutJob.cancel()
                    
                    if (benchmarkResult != null) {
                        val stats = JSONObject()
                        stats.put("duration_ns", benchmarkResult)
                        results.put(name, stats)
                    } else {
                        val errorObj = JSONObject()
                        errorObj.put("error", "timeout or failed")
                        results.put(name, errorObj)
                    }

                } catch (e: Throwable) {
                    Log.e(TAG, "Error running $name benchmark", e)
                    val errorObj = JSONObject()
                    errorObj.put("error", e.message)
                    results.put(name, errorObj)
                }
            }


            
            // Add global duration fields as requested
            if (results.has("CPU") && !results.getJSONObject("CPU").has("error")) {
                reportJson.put("cpu_duration", results.getJSONObject("CPU").getLong("duration_ns"))
            }
            if (results.has("GPU") && !results.getJSONObject("GPU").has("error")) {
                reportJson.put("gpu_duration", results.getJSONObject("GPU").getLong("duration_ns"))
            }
            if (results.has("NPU") && !results.getJSONObject("NPU").has("error")) {
                reportJson.put("npu_duration", results.getJSONObject("NPU").getLong("duration_ns"))
            }

            // Keep the old duration field for backward compatibility (using CPU total)
            if (results.has("CPU") && !results.getJSONObject("CPU").has("error")) {
                reportJson.put("duration", results.getJSONObject("CPU").getLong("duration_ns"))
            } else {
                reportJson.put("duration", -1)
            }

            saveJsonReport(modelFileName, reportJson)
            
            // Give the system time to flush the file to disk before app might be killed
            Thread.sleep(500)
            
            Log.d(TAG, "✅ Benchmark workflow completed successfully for $modelFileName")

            withContext(Dispatchers.Main) {
                val sb = StringBuilder("Benchmark Complete!\n")
                if (results.has("CPU") && !results.getJSONObject("CPU").has("error")) {
                    val cpu = results.getJSONObject("CPU")
                    sb.append("CPU: ${cpu.getLong("duration_ns") / 1_000_000} ms\n")
                }
                if (results.has("GPU") && !results.getJSONObject("GPU").has("error")) {
                    val gpu = results.getJSONObject("GPU")
                    sb.append("GPU: ${gpu.getLong("duration_ns") / 1_000_000} ms\n")
                }
                if (results.has("NPU") && !results.getJSONObject("NPU").has("error")) {
                    val npu = results.getJSONObject("NPU")
                    sb.append("NPU: ${npu.getLong("duration_ns") / 1_000_000} ms\n")
                }
                sb.append("\n✅ Report saved")
                statusText.text = sb.toString()
                progressBar.isVisible = false
            }
        }
    }

    private fun saveJsonReport(modelFileName: String, json: JSONObject) {
        try {
            val cacheDir = externalCacheDir
            if (cacheDir == null) {
                Log.e(TAG, "FATAL: externalCacheDir is null")
                return
            }
            
            val reportFileName = modelFileName.replace(".tflite", ".json")
            val reportFile = File(cacheDir, reportFileName)
            
            // Write with explicit flushing to ensure data is persisted
            reportFile.outputStream().use { outputStream ->
                outputStream.write(json.toString(4).toByteArray())
                outputStream.flush()
            }
            
            // Verify file was written
            if (reportFile.exists() && reportFile.length() > 0) {
                Log.d(TAG, "✅ JSON report saved successfully to: ${reportFile.absolutePath} (${reportFile.length()} bytes)")
            } else {
                Log.e(TAG, "❌ FATAL: Report file not created or empty")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ FATAL: Could not save JSON report: ${e.message}", e)
        }
    }

    private fun decodeResBitmapSafe(@DrawableRes resId: Int, reqMaxDim: Int = 512): Bitmap? {
        try {
            val opt = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeResource(resources, resId, opt)
            val (w, h) = opt.outWidth to opt.outHeight
            if (w <= 0 || h <= 0) return null
            var sample = 1; val maxSide = max(w, h)
            while (maxSide / (sample * 2) >= reqMaxDim) sample *= 2
            val opt2 = BitmapFactory.Options().apply { inSampleSize = sample }
            return BitmapFactory.decodeResource(resources, resId, opt2)
        } catch (_: Throwable) { return null }
    }
}