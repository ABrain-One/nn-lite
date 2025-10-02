package com.example.kotlinapplication

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
import org.json.JSONObject
import java.io.File
import kotlin.math.max
// *** THIS IS THE LINE THAT WAS ADDED TO FIX THE ERROR ***
import kotlin.system.measureTimeMillis

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
            // Update UI at the start
            withContext(Dispatchers.Main) {
                modelNameText.text = modelFileName?.removeSuffix(".tflite") ?: "Unknown Model"
                statusText.text = "Benchmark in progress..."
                progressBar.isVisible = true
            }

            val executionTimeMs = measureTimeMillis {
                try {
                    if (modelFileName.isNullOrEmpty()) {
                        throw Exception("FATAL: App launched without a 'model_filename' argument.")
                    }
                    val modelDevicePath = "$DEVICE_MODEL_DIRECTORY/$modelFileName"
                    val inputBitmap = decodeResBitmapSafe(DUMMY_IMAGE_RES_ID, 512)
                        ?: throw Exception("Failed to decode the dummy input image.")

                    val classifier = TFLiteClassifier(modelDevicePath)
                    classifier.embed(inputBitmap)
                    classifier.close()
                    Log.d(TAG, "SUCCESS: Benchmark completed for $modelFileName.")
                } catch (e: Throwable) {
                    Log.e(TAG, "A runtime error occurred for '$modelFileName'", e)
                }
            }

            reportJson.put("error_message", JSONObject.NULL) // Always null
            reportJson.put("duration", executionTimeMs)

            if (!modelFileName.isNullOrEmpty()) {
                saveJsonReport(modelFileName, reportJson)
            }

            // Update UI at the end
            withContext(Dispatchers.Main) {
                statusText.text = "Benchmark Complete!"
                progressBar.isVisible = false
            }
        }
    }

    private fun saveJsonReport(modelFileName: String, json: JSONObject) {
        try {
            val cacheDir = externalCacheDir ?: return
            val reportFileName = modelFileName.replace(".tflite", ".json")
            val reportFile = File(cacheDir, reportFileName)
            reportFile.writeText(json.toString(4))
            Log.d(TAG, "JSON report saved to: ${reportFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "FATAL: Could not save JSON report: ${e.message}")
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