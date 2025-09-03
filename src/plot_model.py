# make_architecture_diagram.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# --------------------------
# 1) Definizione del modello
# --------------------------
def get_model_paper(input_shape=(256, 256, 3)):
    inputs = keras.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(inputs)
    c1 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(c1)
    c1 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(p1)
    c2 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(c2)
    c2 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(p2)
    c3 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(c3)
    c3 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    c5 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(p3)
    c5 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = layers.Activation("relu")(c5)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(c5)
    c5 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = layers.Activation("relu")(c5)
    p5 = layers.MaxPooling2D(2)(c5)

    c6 = layers.Conv2D(512, 3, kernel_initializer="he_normal", padding="same")(p5)
    c6 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = layers.Activation("relu")(c6)
    c6 = layers.Dropout(0.3)(c6)
    c6 = layers.Conv2D(512, 3, kernel_initializer="he_normal", padding="same")(c6)
    c6 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = layers.Activation("relu")(c6)
    u1 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c6)

    # Decoder
    u6 = layers.Concatenate()([u1, c5])
    c7 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(u6)
    c7 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = layers.Activation("relu")(c7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(c7)
    c7 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = layers.Activation("relu")(c7)
    u2 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c7)

    u7 = layers.Concatenate()([u2, c3])
    c8 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(u7)
    c8 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = layers.Activation("relu")(c8)
    c8 = layers.Dropout(0.2)(c8)
    c8 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(c8)
    c8 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = layers.Activation("relu")(c8)
    u3 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c8)

    u8 = layers.Concatenate()([u3, c2])
    c9 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(u8)
    c9 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = layers.Activation("relu")(c9)
    c9 = layers.Dropout(0.05)(c9)
    c9 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(c9)
    c9 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = layers.Activation("relu")(c9)
    u4 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c9)

    u9 = layers.Concatenate()([u4, c1])
    c10 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(u9)
    c10 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = layers.Activation("relu")(c10)
    c10 = layers.Dropout(0.05)(c10)
    c10 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(c10)
    c10 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = layers.Activation("relu")(c10)

    c11 = layers.Dropout(0.1)(c10)
    c11 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c11)
    c11 = layers.Dropout(0.05)(c11)
    c11 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c11)

    c12 = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c11)
    c12 = layers.Dropout(0.05)(c12)
    c12 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c12)
    c12 = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c12)

    seg_head = layers.Conv2D(3, 1, activation="softmax", name="seg_head")(c12)

    c12_hv = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c11)
    c12_hv = layers.Dropout(0.05)(c12_hv)
    c12_hv = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c12_hv)
    c12_hv = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c12_hv)
    hv_head = layers.Conv2D(2, 1, activation="linear", name="hv_head")(c12_hv)

    model = keras.Model(inputs=inputs, outputs={"seg_head": seg_head, "hv_head": hv_head})
    return model

# --------------------------
# 2) Build del modello
# --------------------------
model = get_model_paper()
model.summary()  # utile in console

# opzionale: salva il summary su file
with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# -----------------------------------------
# 3) Diagramma automatico (SVG e PNG) Keras
# -----------------------------------------
from tensorflow.keras.utils import plot_model

os.makedirs("figures", exist_ok=True)
plot_model(
    model,
    to_file="figures/model_architecture.svg",
    show_shapes=True,          # mostra le shape in output dei layer
    expand_nested=False,       # non serve qui
    dpi=200,
    rankdir="TB"               # top-to-bottom (alternativa: "LR")
)
# versione PNG (se ti serve)
plot_model(
    model,
    to_file="figures/model_architecture.png",
    show_shapes=True,
    expand_nested=False,
    dpi=300,
    rankdir="TB"
)

print("Salvati: figures/model_architecture.svg e .png")

# ------------------------------------------------
# 4) TensorBoard (opzionale) per visualizzare il grafo
# ------------------------------------------------
logdir = "tb_logs"
writer = tf.summary.create_file_writer(logdir)
# traccia una esecuzione per far catturare il grafo
dummy = tf.random.uniform((1, 256, 256, 3))
tf.summary.trace_on(graph=True, profiler=False)
_ = model(dummy, training=False)
with writer.as_default():
    tf.summary.trace_export(name="graph", step=0)
print("TensorBoard log creato in:", logdir)
print("Avvia: tensorboard --logdir tb_logs")

# -----------------------------------------
# 5) Esporta in ONNX e salva .onnx (per Netron)
# -----------------------------------------
import tf2onnx

# firma di input esplicita per ONNX
spec = (tf.TensorSpec((1, 256, 256, 3), tf.float32, name="inputs"),)
onnx_path = "figures/model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,               # 17 o superiore OK
    output_path=onnx_path
)
print("Esportato ONNX in:", onnx_path)

print("\nFatto! 👌  Ora puoi:")
print("  - Usare l'SVG/PNG generato in figures/")
print("  - Aprire model.onnx in Netron (app/website) e fare File -> Export SVG")
