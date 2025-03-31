import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model

# Define input shapes (assuming resized to 224x224)
input_shape = (224, 224, 3)

# Define dual input branches for CC and MLO views
cc_input = Input(shape=input_shape, name='CC_Input')
mlo_input = Input(shape=input_shape, name='MLO_Input')

# EfficientNet feature extractors (shared weights)
efficient_net = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

cc_features = efficient_net(cc_input)
mlo_features = efficient_net(mlo_input)

# Feature Fusion (Concatenation)
fused_features = Concatenate()([cc_features, mlo_features])

# Fully Connected Layers
x = Dense(256, activation='relu')(fused_features)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid', name='Output')  # Binary classification

# Define the model
model = Model(inputs=[cc_input, mlo_input], outputs=out)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

# Print model summary
model.summary()
