# Deep Dive: load_model() & load_config()

This document analyzes the lower-level functions responsible for parsing the model configuration file and using it to instantiate the correct model architecture and load its weights.

## The `load_config()` Function

### Q: What is the single, direct purpose of the `load_config()` function?

**A:** Its only purpose is to **find and parse the `config.json` file** within a model's directory. It takes a file path as input, reads the JSON file, and returns its contents as a standard Python dictionary. It also includes error handling to raise a `FileNotFoundError` if the config file is missing.

---

## The `load_model()` Function

### Q: How does the responsibility of `load_model()` differ from the higher-level `load()` function?

**A:** `load_model()` is a lower-level, specialized function. Its sole responsibility is to take a local path to a model's files and perform the specific steps of **instantiating the neural network object and loading its weights**. Unlike the high-level `load()`, it does not handle downloading from the Hub, loading the tokenizer, or applying LoRA adapters.

### Q: How does the function combine the model's saved configuration with runtime configuration?

**A:** It first calls `load_config()` to get the base configuration dictionary saved with the model. It then uses the `.update()` method to merge the `model_config` dictionary (passed in as an argument) on top of it. This allows a user to override any parameter from the original `config.json` at runtime.

### Q: How does the function find and aggregate the model's weights?

**A:** It uses `glob.glob(str(model_path / "model*.safetensors"))` to find all files in the model's directory that match the pattern "model*.safetensors". This correctly handles models that have their weights split across multiple files. It then iterates through this list, loading each one with `mx.load()` and updating a single `weights` dictionary, effectively merging all shards into one collection.

### Q: Explain the three-step process it uses to dynamically instantiate the correct model architecture.

**A:** It uses a three-step dynamic dispatch process:
1.  **Get Classes:** It calls `get_model_classes(config)` which returns the specific `Model` and `ModelArgs` classes corresponding to the `model_type` in the config.
2.  **Create Args Object:** It calls `model_args_class.from_dict(config)` to create a structured dataclass object from the raw dictionary.
3.  **Instantiate Model:** It finally creates an instance of the model architecture, `model = model_class(model_args)`, passing the structured arguments object to its constructor.

### Q: Describe how post-facto quantization is applied during the loading process.

**A:** After the model object is instantiated but *before* the weights are loaded, the function checks if the `config` dictionary contains a `"quantization"` key. If it does, it means the model weights are saved in a quantized format. It then calls `nn.quantize()`, passing in the model instance and the quantization parameters from the config. This function modifies the model's linear layers in-place, preparing them to accept the quantized weights.

### Q: What is the purpose of the final `model.load_weights(...)` call?

**A:** This is the final and crucial step. After the model object has been created (and potentially quantized), this method takes the large `weights` dictionary and systematically copies each weight tensor into the corresponding parameter of the appropriate layer within the model instance. The `strict=True` argument ensures that the function will fail if there's a mismatch between the weights in the files and the layers in the model architecture.
