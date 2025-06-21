# AI-Powered Code Analysis

[!] AI analysis failed with transformer model sshleifer/distilbart-cnn-12-6: Could not load model sshleifer/distilbart-cnn-12-6 with any of the following classes: (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>, <class 'transformers.models.bart.modeling_bart.BartForConditionalGeneration'>). See the original errors:

while loading with AutoModelForSeq2SeqLM, an error is thrown:
Traceback (most recent call last):
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\pipelines\base.py", line 292, in infer_framework_load_model
    model = model_class.from_pretrained(model, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\models\auto\auto_factory.py", line 571, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 309, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 4574, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 4833, in _load_pretrained_model
    load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 554, in load_state_dict
    check_torch_load_is_safe()
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\utils\import_utils.py", line 1417, in check_torch_load_is_safe
    raise ValueError(
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434

while loading with BartForConditionalGeneration, an error is thrown:
Traceback (most recent call last):
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\pipelines\base.py", line 292, in infer_framework_load_model
    model = model_class.from_pretrained(model, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 309, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 4574, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 4833, in _load_pretrained_model
    load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\modeling_utils.py", line 554, in load_state_dict
    check_torch_load_is_safe()
  File "E:\SCRIPTS\codemancer_tracer\.venv\Lib\site-packages\transformers\utils\import_utils.py", line 1417, in check_torch_load_is_safe
    raise ValueError(
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434


