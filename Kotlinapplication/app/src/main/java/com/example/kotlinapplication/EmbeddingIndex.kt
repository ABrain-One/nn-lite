package com.example.kotlinapplication

import kotlin.math.abs
import kotlin.math.sqrt

/** Tiny in-memory cosine-similarity index for L2-normalized vectors. */
class EmbeddingIndex(private val dim: Int) {
    data class Item(val label: String, val vec: FloatArray)
    private val items = mutableListOf<Item>()

    fun size(): Int = items.size

    /** Add an embedding (L2-normalized or not; we’ll normalize if needed). */
    fun add(label: String, vec: FloatArray) {
        require(vec.size == dim) { "Expected dim=$dim, got ${vec.size}" }
        val v = vec.copyOf()
        val n = l2(v)
        if (!almostOne(n)) { if (n > 0f) for (i in v.indices) v[i] /= n }
        items += Item(label, v)
    }

    /** Return topK nearest by cosine (dot product since vectors are L2-normalized). */
    fun query(query: FloatArray, topK: Int = 5): List<Pair<String, Float>> {
        require(query.size == dim)
        val q = query.copyOf()
        val n = l2(q)
        if (!almostOne(n)) { if (n > 0f) for (i in q.indices) q[i] /= n }

        return items.asSequence()
            .map { it.label to dot(q, it.vec) }
            .sortedByDescending { it.second }
            .take(topK)
            .toList()
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
    private fun l2(v: FloatArray): Float {
        var s = 0f
        for (x in v) s += x * x
        return sqrt(s)
    }
    private fun almostOne(x: Float, eps: Float = 1e-3f) = abs(x - 1f) <= eps
}
