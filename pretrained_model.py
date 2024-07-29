import tensorflow as tf

def save_pretrained_model(model_save_path):
    # Load a pretrained EfficientNet model
    model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    
    # Save the model
    model.save(model_save_path)

model_save_path = 'efficientnet_tf.h5'
save_pretrained_model(model_save_path)
