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

                    // Timeout: 120s for CPU, 90s for GPU/NPU (increased for multiple iterations)
                    val timeoutMs = if (type == TFLiteClassifier.DelegateType.CPU) 120000L else 90000L
                    
                    // Run benchmark with timeout using CompletableDeferred
                    val result = CompletableDeferred<JSONObject?>()
                    
                    // Launch blocking work in a separate coroutine
                    val job = launch(Dispatchers.Default) {
                        try {
                            val classifier = TFLiteClassifier(modelDevicePath, delegateType = type)
                            classifier.embed(inputBitmap) // Warmup
                            
                            val iterations = 25
                            val durations = LongArray(iterations)
                            var successCount = 0
                            
                            for (i in 0 until iterations) {
                                val startTime = System.nanoTime()
                                classifier.embed(inputBitmap)
                                val duration = System.nanoTime() - startTime
                                durations[i] = duration
                                successCount++
                            }
                            
                            classifier.close()
                            
                            if (successCount > 0) {
                                val min = durations.minOrNull() ?: 0L
                                val max = durations.maxOrNull() ?: 0L
                                val avg = durations.average()
                                
                                // Calculate Standard Deviation
                                var sumSqDiff = 0.0
                                for (d in durations) {
                                    sumSqDiff += Math.pow(d - avg, 2.0)
                                }
                                val stdDev = Math.sqrt(sumSqDiff / durations.size)
                                
                                val stats = JSONObject()
                                stats.put("duration_ns", avg.toLong()) // Primary is now Average
                                stats.put("min_ns", min)
                                stats.put("max_ns", max)
                                stats.put("avg_ns", avg)
                                stats.put("std_dev_ns", stdDev)
                                stats.put("iterations", successCount)
                                
                                Log.d(TAG, "SUCCESS: $name Benchmark completed. Avg: ${avg.toLong()}ns, StdDev: ${stdDev.toLong()}ns")
                                result.complete(stats)
                            } else {
                                result.complete(null)
                            }
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
                        results.put(name, benchmarkResult)
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


            
            // Add global duration fields as requested (using Average)
            if (results.has("CPU") && !results.getJSONObject("CPU").has("error")) {
                val cpuStats = results.getJSONObject("CPU")
                reportJson.put("cpu_duration", cpuStats.getLong("avg_ns")) // Average
                reportJson.put("cpu_min_duration", cpuStats.getLong("min_ns"))
                reportJson.put("cpu_max_duration", cpuStats.getLong("max_ns"))
                reportJson.put("cpu_std_dev", cpuStats.getDouble("std_dev_ns"))
                reportJson.put("iterations", cpuStats.getInt("iterations"))
            }
            if (results.has("GPU") && !results.getJSONObject("GPU").has("error")) {
                val gpuStats = results.getJSONObject("GPU")
                reportJson.put("gpu_duration", gpuStats.getLong("avg_ns")) // Average
                reportJson.put("gpu_min_duration", gpuStats.getLong("min_ns"))
                reportJson.put("gpu_max_duration", gpuStats.getLong("max_ns"))
                reportJson.put("gpu_std_dev", gpuStats.getDouble("std_dev_ns"))
            }
            if (results.has("NPU") && !results.getJSONObject("NPU").has("error")) {
                val npuStats = results.getJSONObject("NPU")
                reportJson.put("npu_duration", npuStats.getLong("avg_ns")) // Average
                reportJson.put("npu_min_duration", npuStats.getLong("min_ns"))
                reportJson.put("npu_max_duration", npuStats.getLong("max_ns"))
                reportJson.put("npu_std_dev", npuStats.getDouble("std_dev_ns"))
            }

            // Calculate the best (minimum) average duration across all successful delegates
            var bestAvgDuration = Long.MAX_VALUE
            var bestUnit = "Unknown"
            var hasValidDuration = false

            // Helper to get hardware names
            val cpuName = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
            val gpuName = try {
                // Quick hack to get GPU renderer without full GLSurfaceView
                val egl = javax.microedition.khronos.egl.EGLContext.getEGL() as javax.microedition.khronos.egl.EGL10
                val dpy = egl.eglGetDisplay(javax.microedition.khronos.egl.EGL10.EGL_DEFAULT_DISPLAY)
                val version = IntArray(2)
                egl.eglInitialize(dpy, version)
                val configAttribs = intArrayOf(
                    javax.microedition.khronos.egl.EGL10.EGL_RENDERABLE_TYPE, 4, // EGL_OPENGL_ES2_BIT
                    javax.microedition.khronos.egl.EGL10.EGL_SURFACE_TYPE, javax.microedition.khronos.egl.EGL10.EGL_PBUFFER_BIT,
                    javax.microedition.khronos.egl.EGL10.EGL_NONE
                )
                val configs = arrayOfNulls<javax.microedition.khronos.egl.EGLConfig>(1)
                val numConfigs = IntArray(1)
                egl.eglChooseConfig(dpy, configAttribs, configs, 1, numConfigs)
                val ctxAttribs = intArrayOf(0x3098, 2, javax.microedition.khronos.egl.EGL10.EGL_NONE) // EGL_CONTEXT_CLIENT_VERSION
                val ctx = egl.eglCreateContext(dpy, configs[0], javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT, ctxAttribs)
                val surfAttribs = intArrayOf(javax.microedition.khronos.egl.EGL10.EGL_WIDTH, 1, javax.microedition.khronos.egl.EGL10.EGL_HEIGHT, 1, javax.microedition.khronos.egl.EGL10.EGL_NONE)
                val surf = egl.eglCreatePbufferSurface(dpy, configs[0], surfAttribs)
                egl.eglMakeCurrent(dpy, surf, surf, ctx)
                val renderer = android.opengl.GLES20.glGetString(android.opengl.GLES20.GL_RENDERER)
                egl.eglMakeCurrent(dpy, javax.microedition.khronos.egl.EGL10.EGL_NO_SURFACE, javax.microedition.khronos.egl.EGL10.EGL_NO_SURFACE, javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT)
                egl.eglDestroySurface(dpy, surf)
                egl.eglDestroyContext(dpy, ctx)
                egl.eglTerminate(dpy)
                renderer ?: "GPU"
            } catch (e: Exception) {
                "GPU"
            }
            val npuName = "$cpuName NPU" // Best effort for NPU name

            if (results.has("CPU") && !results.getJSONObject("CPU").has("error")) {
                val duration = results.getJSONObject("CPU").getLong("avg_ns")
                if (duration < bestAvgDuration) {
                    bestAvgDuration = duration
                    bestUnit = cpuName
                }
                hasValidDuration = true
            }
            if (results.has("GPU") && !results.getJSONObject("GPU").has("error")) {
                val duration = results.getJSONObject("GPU").getLong("avg_ns")
                if (duration < bestAvgDuration) {
                    bestAvgDuration = duration
                    bestUnit = gpuName
                }
                hasValidDuration = true
            }
            if (results.has("NPU") && !results.getJSONObject("NPU").has("error")) {
                val duration = results.getJSONObject("NPU").getLong("avg_ns")
                if (duration < bestAvgDuration) {
                    bestAvgDuration = duration
                    bestUnit = npuName
                }
                hasValidDuration = true
            }

            if (hasValidDuration) {
                reportJson.put("duration", bestAvgDuration)
                reportJson.put("unit", bestUnit)
            } else {
                reportJson.put("duration", -1)
                reportJson.put("unit", "None")
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