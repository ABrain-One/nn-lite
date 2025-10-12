plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.App"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.App"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += ""
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions { jvmTarget = "11" }

    // Keep .tflite files uncompressed
    aaptOptions { noCompress("tflite") }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

/* ---- Remove LiteRT globally to avoid duplicate classes ---- */
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    exclude(group = "com.google.ai.edge.litert", module = "litert")
}

dependencies {
    dependencies {
        implementation(libs.androidx.core.ktx)
        implementation(libs.androidx.appcompat)
        implementation(libs.material)
        implementation(libs.androidx.activity)
        implementation(libs.androidx.constraintlayout)
        testImplementation(libs.junit)
        androidTestImplementation(libs.androidx.junit)
        androidTestImplementation(libs.androidx.espresso.core)

        // ✅ Stable combo
        implementation("org.tensorflow:tensorflow-lite:2.12.0")
        implementation("org.tensorflow:tensorflow-lite-support:0.4.3")
        implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
        implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.12.0")


        // Add only if you get "Op not found" runtime errors:
    // implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.17.0")

    // Optional GPU delegate (enable after CPU path works):
    // implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
}
}

