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
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private var classifier: TFLiteClassifier? = null

    private lateinit var ivQuery: ImageView
    private lateinit var tv: TextView
    private lateinit var btn: Button
    private lateinit var progress: ProgressBar

    private lateinit var nearestTitle: TextView
    private lateinit var nearestPreview: ImageView
    private lateinit var nearestCaption: TextView

    // Your embedding model in assets/
    private val modelFile = "dpn107_model_0.1204.tflite"

    // 🔹 Your drawables in res/drawable
    private val galleryResIds: List<Int> = listOf(
        R.drawable.car,
        R.drawable.cat,
        R.drawable.dog,
        R.drawable.su30,
        R.drawable.su34
    )

    private val index = EmbeddingIndex(dim = 512)
    private val labelToRes = mutableMapOf<String, Int>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivQuery = findViewById(R.id.imagePreview)
        tv = findViewById(R.id.resultText)
        btn = findViewById(R.id.runButton)
        progress = findViewById(R.id.progress)

        nearestTitle = findViewById(R.id.nearestTitle)
        nearestPreview = findViewById(R.id.nearestPreview)
        nearestCaption = findViewById(R.id.nearestCaption)

        // Use the first gallery image as the query image
        val queryResId: Int = galleryResIds.first()
        val queryBmp = decodeResBitmapSafe(queryResId, reqMaxDim = 512)
            ?: run { tv.text = "Failed to decode query image."; btn.isEnabled = false; return }

        ivQuery.setImageBitmap(queryBmp)

        // Init model
        try {
            classifier = TFLiteClassifier(
                context = this,
                modelPath = modelFile,
                numThreads = 1,
                useNNAPI = false
            )
            tv.text = "Model loaded (embedding mode)."
        } catch (e: Throwable) {
            tv.text = "Model init failed: ${e.message}"
            btn.isEnabled = false
            return
        }

        // Build the gallery index — IMPORTANT: skip the query image
        lifecycleScope.launch {
            progress.isVisible = true
            tv.append("\nBuilding gallery index…")
            val added = withContext(Dispatchers.Default) {
                var count = 0
                val c = classifier!!
                for (resId in galleryResIds.filter { it != queryResId }) {
                    try {
                        decodeResBitmapSafe(resId, reqMaxDim = 512)?.let { bmp ->
                            val vec = c.embed(bmp) // L2-normalized 512-d
                            val label = labelFor(resId)
                            index.add(label, vec)
                            labelToRes[label] = resId
                            bmp.recycle()
                            count++
                        }
                    } catch (t: Throwable) {
                        Log.e("MainActivity", "Index add failed for ${labelFor(resId)}", t)
                    }
                }
                count
            }
            tv.append("\nIndex ready. Items=$added")
            progress.isVisible = false
        }

        btn.setOnClickListener {
            val c = classifier ?: return@setOnClickListener
            progress.isVisible = true
            btn.isEnabled = false
            tv.append("\nRunning inference…")

            lifecycleScope.launch {
                try {
                    val (vec, top) = withContext(Dispatchers.Default) {
                        val v = c.embed(queryBmp)
                        val matches = if (index.size() > 0) index.query(v, topK = 5) else emptyList()
                        v to matches
                    }

                    val preview = vec.take(8).joinToString(", ") { "%.4f".format(it) }
                    val knnText = buildString {
                        appendLine("\nNearest in gallery (cosine):")
                        top.forEachIndexed { i, (label, cos) ->
                            appendLine("${i + 1}. $label   cos=${"%.4f".format(cos)}")
                        }
                    }
                    tv.append("\nEmbedding L2-normed, dim=${vec.size}\nfirst 8: [ $preview ]$knnText")

                    // Show top-1 nearest (now it cannot be the same image)
                    if (top.isNotEmpty()) {
                        val (bestLabel, bestCos) = top.first()
                        val resId = labelToRes[bestLabel]
                        nearestTitle.isVisible = true
                        nearestCaption.isVisible = true
                        if (resId != null) {
                            nearestPreview.isVisible = true
                            nearestPreview.setImageResource(resId)
                            nearestCaption.text = "$bestLabel   cos=${"%.4f".format(bestCos)}"
                        } else {
                            nearestPreview.isVisible = false
                            nearestCaption.text = "Nearest: $bestLabel   cos=${"%.4f".format(bestCos)} (no drawable found)"
                        }
                    } else {
                        nearestTitle.isVisible = false
                        nearestPreview.isVisible = false
                        nearestCaption.isVisible = true
                        nearestCaption.text = "No nearest result."
                    }
                } finally {
                    progress.isVisible = false
                    btn.isEnabled = true
                }
            }
        }
    }

    private fun labelFor(@DrawableRes resId: Int): String =
        runCatching { resources.getResourceEntryName(resId) }.getOrElse { "res_$resId" }

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

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
