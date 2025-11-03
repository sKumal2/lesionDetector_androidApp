package com.example.skinlesionvit

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var model: Module
    private lateinit var imagePreview: ImageView
    private lateinit var resultText: TextView

    /* --------------------------------------------------------------------- *
     *  Image picker – only gallery (you can add camera later if you want)
     * --------------------------------------------------------------------- */
    private val pickImage =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val uri: Uri? = result.data?.data
                uri?.let {
                    val bitmap = uriToBitmap(it) ?: return@let
                    imagePreview.setImageBitmap(bitmap)
                    predictLesion(bitmap)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imagePreview = findViewById(R.id.imagePreview)
        resultText = findViewById(R.id.resultText)
        val uploadButton: Button = findViewById(R.id.uploadButton)

        uploadButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            pickImage.launch(intent)
        }

        // ------------------- Load the PyTorch Lite model -------------------
        try {
            model = LiteModuleLoader.load(assetFilePath("lesion_model.ptl"))
        } catch (e: IOException) {
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    /* --------------------------------------------------------------------- *
     *  Convert a content Uri → Bitmap (handles large images safely)
     * --------------------------------------------------------------------- */
    private fun uriToBitmap(uri: Uri): Bitmap? = try {
        contentResolver.openInputStream(uri)?.use { input ->
            BitmapFactory.decodeStream(input)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }

    /* --------------------------------------------------------------------- *
     *  Main inference entry point – runs on a background thread
     * --------------------------------------------------------------------- */
    private fun predictLesion(bitmap: Bitmap) {
        resultText.text = "Analyzing…"
        thread {
            try {
                val inputTensor = bitmapToInputTensor(bitmap)
                val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
                val scores = outputTensor.dataAsFloatArray

                // ---- adapt to your model output (2 classes in your original code) ----
                val noLesionProb = scores[0]
                val lesionProb = scores[1]

                val result = if (lesionProb > noLesionProb) {
                    "Lesion detected (prob: ${"%.1f".format(lesionProb * 100)}%)"
                } else {
                    "No lesion (prob: ${"%.1f".format(noLesionProb * 100)}%)"
                }

                runOnUiThread { resultText.text = result }
            } catch (e: Exception) {
                runOnUiThread {
                    resultText.text = "Error: ${e.message}"
                    Toast.makeText(this@MainActivity, "Inference failed", Toast.LENGTH_SHORT).show()
                }
                e.printStackTrace()
            }
        }
    }

    /* --------------------------------------------------------------------- *
     *  PRE-PROCESSING
     *  – resize to 256×256 (any square size ≥ 128 works)
     *  – extract **exactly 128** random 16×16 patches
     *  – keep only R & G channels (2 channels → shape [1,128,16,2,16,2])
     *  – ImageNet normalisation
     * --------------------------------------------------------------------- */
    private fun bitmapToInputTensor(original: Bitmap): Tensor {
        // 1. Resize to a square that comfortably holds 128 patches
        val IMG_SIZE = 256
        val PATCH_SIZE = 16
        val NUM_PATCHES = 128
        val CHANNELS = 2          // R, G  (drop B)

        val resized = Bitmap.createScaledBitmap(original, IMG_SIZE, IMG_SIZE, true)

        // Allocate exactly the size the model expects
        val buffer = FloatBuffer.allocate(1 * NUM_PATCHES * PATCH_SIZE * CHANNELS * PATCH_SIZE * 2) // 131072
        buffer.rewind()

        val pixels = IntArray(PATCH_SIZE * PATCH_SIZE)
        val rnd = java.util.Random()

        // ImageNet mean / std for the two channels we keep
        val mean = floatArrayOf(0.485f, 0.456f)          // R, G
        val std  = floatArrayOf(0.229f, 0.224f)          // R, G

        repeat(NUM_PATCHES) {
            // pick a random top-left corner inside the image
            val x = rnd.nextInt(IMG_SIZE - PATCH_SIZE)
            val y = rnd.nextInt(IMG_SIZE - PATCH_SIZE)

            resized.getPixels(pixels, 0, PATCH_SIZE, x, y, PATCH_SIZE, PATCH_SIZE)

            for (pixel in pixels) {
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8)  and 0xFF) / 255f

                buffer.put((r - mean[0]) / std[0])
                buffer.put((g - mean[1]) / std[1])
            }
        }

        buffer.rewind()
        return Tensor.fromBlob(buffer, longArrayOf(1, 128, 16, 2, 16, 2))
    }

    /* --------------------------------------------------------------------- *
     *  Helper – copy model from assets to internal files (first run only)
     * --------------------------------------------------------------------- */
    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buf = ByteArray(4 * 1024)
                var len: Int
                while (input.read(buf).also { len = it } != -1) {
                    output.write(buf, 0, len)
                }
                output.flush()
            }
        }
        return file.absolutePath
    }
}