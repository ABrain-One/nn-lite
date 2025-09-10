package com.example.kotlinapplication

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
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
import kotlin.system.measureTimeMillis

class MainActivity : AppCompatActivity() {

    private lateinit var ivQuery: ImageView
    private lateinit var tv: TextView
    private lateinit var btn: Button
    private lateinit var progress: ProgressBar

    private val galleryResIds: List<Int> = listOf(
        R.drawable.car,
        R.drawable.cat,
        R.drawable.dog,
        R.drawable.su30,
        R.drawable.su34
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivQuery = findViewById(R.id.imagePreview)
        tv = findViewById(R.id.resultText)
        btn = findViewById(R.id.runButton)
        progress = findViewById(R.id.progress)

        val queryResId = galleryResIds.first()
        val queryBmp = decodeResBitmapSafe(queryResId, 512)
            ?: run { tv.text = "Failed to decode query image."; btn.isEnabled = false; return }

        ivQuery.setImageBitmap(queryBmp)

        btn.setOnClickListener {
            btn.isEnabled = false
            tv.text = ""
            progress.isVisible = true

            lifecycleScope.launch {
                val modelFiles = assets.list("")?.filter { it.endsWith(".tflite") } ?: emptyList()
                for (modelName in modelFiles) {
                    try {
                        val duration = measureTimeMillis {
                            val classifier = TFLiteClassifier(this@MainActivity, modelName)
                            classifier.embed(queryBmp)
                            classifier.close()
                        }

                        // Create JSON for this model
                        val json = JSONObject().apply {
                            put("model_name", modelName)
                            put("duration_ms", duration)
                        }

                        val jsonFileName = "${modelName.removeSuffix(".tflite")}.json"
                        openFileOutput(jsonFileName, MODE_PRIVATE).use { it.write(json.toString().toByteArray()) }

                        // Update UI
                        tv.append("\nmodel name: $modelName")
                        tv.append("\nsuccessfully executed: Yes")
                        tv.append("\nJSON file has been created successfully\n")

                    } catch (e: Throwable) {
                        Log.e("MainActivity", "Error executing $modelName", e)
                        tv.append("\nmodel name: $modelName")
                        tv.append("\nsuccessfully executed: No")
                        tv.append("\nError: ${e.message}\n")
                    }
                }
                progress.isVisible = false
                btn.isEnabled = true
            }
        }
    }

    private fun decodeResBitmapSafe(@DrawableRes resId: Int, reqMaxDim: Int = 512): Bitmap? {
        return try {
            val opt = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeResource(resources, resId, opt)
            val (w, h) = opt.outWidth to opt.outHeight
            if (w <= 0 || h <= 0) return null
            var sample = 1
            var maxSide = max(w, h)
            while (maxSide / (sample * 2) >= reqMaxDim) sample *= 2
            val opt2 = BitmapFactory.Options().apply { inSampleSize = sample }
            BitmapFactory.decodeResource(resources, resId, opt2)
        } catch (_: Throwable) { null }
    }
}
// Create JSON for this model
