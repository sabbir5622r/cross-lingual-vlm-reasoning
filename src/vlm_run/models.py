import gc
import time

from .storage import environment


class Runner:
    def __init__(self, details, revision):
        self.details = details
        self.revision = revision
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for model inference")
        precision = self.details["precision"]
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This model configuration requires BF16. Use a supporting lab GPU or create an explicitly named FP32 configuration; precision is not changed silently.")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        family = self.details["family"]
        if family == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration

            cls = Qwen2_5_VLForConditionalGeneration
        elif family == "gemma":
            from transformers import Gemma3ForConditionalGeneration

            cls = Gemma3ForConditionalGeneration
        else:
            raise ValueError(f"Unknown model family: {family}")
        processor_options = {}
        if family == "qwen":
            processor_options = {key: self.details[key] for key in ["min_pixels", "max_pixels"]}
        self.processor = AutoProcessor.from_pretrained(self.details["model_id"], revision=self.revision, **processor_options)
        self.model = cls.from_pretrained(
            self.details["model_id"], revision=self.revision,
            torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
        )
        devices = set(str(value) for value in getattr(self.model, "hf_device_map", {}).values())
        if devices.intersection({"cpu", "disk"}):
            self.close()
            raise RuntimeError("Model was offloaded to CPU/disk. Use more GPU memory to keep inference comparisons consistent.")
        self.model.eval()
        self.devices = [index for index in range(torch.cuda.device_count())]
        for index in self.devices:
            torch.cuda.reset_peak_memory_stats(index)
        return self

    def prepare(self, image, question, instruction, options=None):
        prompt = f"{instruction}\nQuestion: {question}"
        if options:
            prompt += f"\nOptions: {options}"
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs)
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        import torch

        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(dtype=self.model.dtype)
        return inputs, prompt

    def synchronize(self):
        import torch

        for index in self.devices:
            torch.cuda.synchronize(index)

    def generate(self, inputs, maximum):
        import torch

        self.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=maximum, do_sample=False)
        self.synchronize()
        elapsed = time.perf_counter() - started
        tokens = output[0, inputs["input_ids"].shape[-1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.processor.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        ended = bool(tokens and tokens[-1] in eos_ids)
        text = self.processor.batch_decode([tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return {
            "prediction": text, "generation_seconds": elapsed,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": len(tokens), "ended_with_eos": ended,
            "generation_limit_reached": len(tokens) >= maximum and not ended,
        }

    def memory(self):
        import torch

        return [
            {"device": index, "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 1024**3,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 1024**3}
            for index in self.devices
        ]

    def close(self):
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
