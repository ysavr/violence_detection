# convert_to_tflite.py
import tensorflow as tf

def convert_to_tflite(saved_model_dir, tflite_model_path="violence_lstm_model_sample.tflite"):
    """
    Convert a SavedModel directory to TFLite format.
    
    Args:
        saved_model_dir (str): Path to the SavedModel directory.
        tflite_model_path (str): Path to save the .tflite model file (default: 'violence_lstm_model.tflite').
    """
    print("Starting TFLite conversion process...")
    print("TensorFlow version:", tf.__version__)

    # Create the TFLite converter from SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable resource variables for LSTM compatibility
    converter.experimental_enable_resource_variables = True

    # Allow SELECT_TF_OPS and disable tensor list lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    # Optimize for size and performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Ensure float32 inputs/outputs
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # Define input shape explicitly for TFLite
    input_shape = (1, 10, 51)  # Batch size 1, 10 timesteps, 51 features
    input_tensor_name = 'lstm_input'  # Match the name from training
    converter.input_shapes = {input_tensor_name: input_shape}

    # Convert the model
    try:
        tflite_model = converter.convert()
        print("Model successfully converted to TFLite format.")
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise

    # Save the .tflite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model successfully saved as: {tflite_model_path}")

    # --- VERIFICATION ---
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    print("TFLite model successfully loaded and tensors allocated.")

    # --- MODEL INFO ---
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Details:", input_details)
    print("Output Details:", output_details)

if __name__ == "__main__":
    # Path to your SavedModel directory
    saved_model_dir = "violence_lstm_saved_model"
    
    # Run the conversion
    convert_to_tflite(saved_model_dir)