# Function and Import Summary

## setup_assistant

**Functions:** yes_no_prompt(prompt_text), load_checkpoints(), save_checkpoints(checkpoints), main(), lazy_create_dataset(dataset_path), lazy_fine_tune_model(dataset_path, model_save_path)

**Imports:** TTS.tts.models.xtts, torch, importlib, TTS.api, modules.config, modules.install_dependencies, modules.download_and_models, modules.dataset, sys, modules.db_setup, dotenv, TTS.config.shared_configs, modules.device_detector, modules.utils, modules.api_key_setup, subprocess, os, TTS.tts.configs.xtts_config, modules.model_training, sounddevice, json

## voice_assistant

**Functions:** task_name(task), on_wakeword_detected(), run_assistant()

**Imports:** modules.intent_logic, modules.contractions, threading, typing, modules.db_manager, warnings, logging, modules.intent_classifier, asyncio, modules.tts_service, modules.audio_utils, nest_asyncio, wakeword_detector, modules.weather_service, platform, modules.stt_service, modules.file_watcher_service, modules.llm_service

## wakeword_detector

**Functions:** get_porcupine_key(), test_callback()

**Imports:** logging, asyncio, pvporcupine, modules.config, precise_runner, sounddevice, os

## api_key_setup

**Functions:** setup_api_key(key_file_path, service_name, prompt_message)

**Imports:** typing, getpass, os

## audio_utils

**Functions:** record_audio(sample_rate, duration)

**Imports:** sounddevice, numpy, config, asyncio

## calendar_utils

**Functions:** add_event_to_calendar(summary, start, end, description), get_calendar_file_path()

**Imports:** ics, datetime, typing, os

## config

**Functions:** get_picovoice_key(), get_openweather_api_key()

**Imports:** logging, dotenv, os

## config_env

**Functions:** None

**Imports:** dotenv, os

## contractions

**Functions:** load_json_map(filename), load_word_list_from_file(filename), _load_and_compile_normalization_data(), reload_normalization_data(), normalize_text(text), replace_contraction(match), replace_misspelling(match)

**Imports:** logging, spellchecker, re, json, os

## dataset

**Functions:** create_dataset(dataset_path)

**Imports:** pandas

## db_manager

**Functions:** initialize_db(), _save(), _fetch()

**Imports:** datetime, sqlite3, config, asyncio

## db_setup

**Functions:** setup_db(DB_PATH)

**Imports:** sqlite3

## device_detector

**Functions:** get_cpu_info(), get_gpu_info(), detect_cuda_with_torch(), get_cuda_device_name_with_torch(), recommend_torch_version(), detect_cpu_vendor(), write_env_file(settings, base_dir_path, asr_device, tts_device), install_pytorch_version(torch_version_str), run_device_setup(base_dir_path_str)

**Imports:** GPUtil, torch, sys, logging, subprocess, pathlib, platform

## download_and_models

**Functions:** download_file(url, dest), play_sample_tts(model_name_for_tts, speed_rate, sample_text), setup_tts(), setup_precise(base_dir, model_url), setup_stt_model()

**Imports:** TTS.tts.models.xtts, config, torch, modules.stt_model_selection, sys, modules.whisperx_setup, urllib.request, subprocess, dotenv, TTS.config.shared_configs, modules.whisper_setup, TTS.tts.configs.xtts_config, TTS.api, sounddevice, os

## error_handling

**Functions:** async_error_handler(timeout), decorator(func)

**Imports:** typing, functools, logging, asyncio

## file_watcher_service

**Functions:** __init__(self, reload_callback, files_to_watch), on_any_event(self, event), start_normalization_data_watcher(reload_callback, data_dir, filenames)

**Imports:** typing, logging, time, watchdog.observers, watchdog.events, os

## greeting_module

**Functions:** _load_variations(label), get_greeting(), get_goodbye()

**Imports:** pandas, random, os

## gui_utils

**Functions:** show_reminders_gui(reminders, date_str)

**Imports:** tkinter, datetime

## install_dependencies

**Functions:** check_prerequisites(), run_command(cmd, error_msg), install_system_dependencies(), install_python_dependencies(), list_ollama_models(), pull_ollama_model(), test_tts_installation(), install_dependencies()

**Imports:** sys, TTS.api, logging, subprocess, dotenv, modules.config, shutil, platform, os

## intent_classifier

**Functions:** initialize_intent_classifier()

**Imports:** config, torch, typing, os, asyncio, pandas, joint_model, transformers

## intent_logic

**Functions:** intent_handler(intent_name), decorator(func), get_response(intent_key)

**Imports:** modules.contractions, modules.calendar_utils, modules.weather_service, modules.stt_service, modules.config, threading, scripts.intent_validator, asyncio, modules.tts_service, modules.audio_utils, modules.retrain_utils, modules.db_manager, pandas, re, modules.error_handling, modules.reminder_utils, os, dateparser, typing, logging, modules.intent_classifier, datetime, modules.gui_utils, json, modules.llm_service

## joint_model

**Functions:** __init__(self, config), forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict, intent_labels, slot_labels)

**Imports:** torch, typing, dataclasses, transformers.utils, transformers

## llm_service

**Functions:** get_session_history(session_id), initialize_llm(), get_llm_response_sync(input_text)

**Imports:** langchain_core.prompts, config, typing, langchain_core.runnables.history, langchain_core.runnables.config, langchain_community.llms, asyncio, langchain_core.chat_history, langchain_community.chat_message_histories

## model_training

**Functions:** fine_tune_model(dataset_path, model_save_path), process_data_for_joint_model(examples)

**Imports:** modules.contractions, transformers, torch, typing, sys, scripts.intent_validator, subprocess, re, datasets, pandas, modules.joint_model, argparse, json, os

## ollama_setup

**Functions:** None

**Imports:** None

## reminder_utils

**Functions:** _parse_time_from_entities_text(time_str, date_ref_str, now), _parse_date_from_entities_text(date_ref_str), parse_reminder(text, entities), parse_list_reminder_request(text, entities)

**Imports:** typing, re, logging, datetime

## retrain_utils

**Functions:** parse_retrain_request(text)

**Imports:** sys, subprocess, os, asyncio

## stt_model_selection

**Functions:** describe_stt_models(), prompt_stt_model_choice()

**Imports:** None

## stt_service

**Functions:** initialize_stt(), transcribe_audio(audio_data_np_int16)

**Imports:** whisperx, numpy, config, asyncio

## tts_service

**Functions:** initialize_tts(), text_to_speech(text)

**Imports:** TTS.tts.models.xtts, torch, config, TTS.config.shared_configs, asyncio, TTS.tts.configs.xtts_config, TTS.api, sounddevice, os

## utils

**Functions:** create_directories(BASE_DIR, MODEL_SAVE_PATH)

**Imports:** os

## weather_service

**Functions:** initialize_weather_service()

**Imports:** config, typing, logging, asyncio, aiohttp, os

## whisperx_setup

**Functions:** get_project_root(), setup_whisperx()

**Imports:** tempfile, sys, scipy.io.wavfile, subprocess, dotenv, whisperx, modules.config, sounddevice, os

## whisper_setup

**Functions:** load_whisper_model(model_size, device), transcribe_with_whisper(audio_np, model_size, device), get_whisper_model_descriptions(), setup_whisper()

**Imports:** numpy, tempfile, scipy.io.wavfile, dotenv, whisper, modules.config, sounddevice, os

## __init__

**Functions:** None

**Imports:** None

## device_manager

**Functions:** load_devices(config_path), get_device(name), _save_devices_and_update_cache(devices_to_save, config_path), list_devices(), announce_device_info(device_name), add_device(name, mac_address, ip_address, group, type), confirm_action(prompt), remove_device(name), update_device(name), _ping_ip(ip_address, timeout), check_all_device_statuses(), get_devices_by_group(group), get_devices_by_type(type_filter), list_devices_by_type(type_), wake_group(group), ping_group(group), register_intents(), auto_discover_new_devices(interval_minutes), discovery_loop()

**Imports:** threading, typing, logging, subprocess, time, modules.find_devices, modules.wol, platform, core.tts, json, os

## find_devices

**Functions:** get_local_ip(), find_devices(), announce_known_devices(), find_and_suggest_new_devices(), register_intents()

**Imports:** modules.device_manager, logging, subprocess, socket, core.tts

## general

**Functions:** tell_time(), run_self_test(), log_and_speak(message, level), hello(), register_intents()

**Imports:** sys, logging, subprocess, time, core.tts

## ping

**Functions:** _is_valid_ip_format(ip_string), ping_target(target_identifier), register_intents()

**Imports:** modules.device_manager, logging, subprocess, platform, core.tts, os

## server

**Functions:** boot_system(system_name), start_server(server_name), stop_server(), register_intents()

**Imports:** modules.device_manager, typing, logging, time, modules.wol, core.tts, modules.ping, os

## shutdown

**Functions:** _perform_shutdown(), request_shutdown_confirmation(argument), request_assistant_exit(), register_intents()

**Imports:** typing, logging, time, platform, core.tts, os

## speedtest

**Functions:** run_speedtest(), register_intents()

**Imports:** speedtest, core.tts, logging

## system_info

**Functions:** bytes_to_gb(bytes_val), format_uptime(seconds), get_cpu_usage_speak(), get_memory_usage_speak(), get_disk_usage_speak(path_argument), get_system_uptime_speak(), get_system_summary_speak(), get_cpu_load_speak(), register_intents()

**Imports:** typing, logging, time, datetime, platform, core.tts, psutil, os

## weather

**Functions:** get_weather_wmo_description(code), get_weather(city_name), register_intents()

**Imports:** requests, logging, core.tts, json, os

## wol

**Functions:** is_valid_mac(mac), load_systems_config(config_path), send_wol_packet(mac_address, tts), wake_on_lan(device_name), register_intents()

**Imports:** modules.device_manager, typing, logging, socket, re, core.tts, json, os

## __init__

**Functions:** None

**Imports:** None

## archive_augmented_data

**Functions:** archive_augmented_files()

**Imports:** shutil, datetime, os

## augment_dictionaries

**Functions:** augment_dictionary(orig_data), main()

**Imports:** json, pathlib

## augment_intent_dataset

**Functions:** edit_distance(s1, s2), archive_augmented_data_before_augmentation(), synonym_replace(text), model_paraphrase(text), generate_paraphrases(text)

**Imports:** modules.contractions, transformers, torch, sys, subprocess, nltk, tqdm, pandas, nltk.corpus, spellchecker, modules.device_detector, collections, json, os

## intent_validator

**Functions:** validate_intents()

**Imports:** modules.retrain_utils, typing, sys, pandas, os

## validate_and_clean_sentences

**Functions:** clean_sentence(text), main()

**Imports:** tqdm, pandas, argparse, language_tool_python, os

## __init__

**Functions:** None

**Imports:** None

## test_intent_classifier

**Functions:** sample_csv_data(), mock_model_path(tmp_path), mock_tokenizer(), mock_joint_model(), reset_global_state()

**Imports:** torch, unittest.mock, sys, modules.intent_classifier, pandas, modules.joint_model, pytest, os

## test_intent_logic

**Functions:** mock_response_map(), mock_entities(), setup_and_teardown(), test_intent_handler_decorator_registers_function(self), test_intent_handler_decorator_returns_original_function(self), test_multiple_intent_handlers_registered(self), test_get_response_basic_lookup(self, mock_response_map), test_get_response_with_formatting(self, mock_response_map), test_get_response_missing_key_in_format(self, mock_response_map), test_get_response_formatting_error(self, mock_response_map), test_get_response_nonexistent_intent(self, mock_response_map), test_intent_handlers_registered(self, intent, expected_handler)

**Imports:** modules.intent_logic, numpy, unittest.mock, sys, pytest, datetime, os

## test_joint_model

**Functions:** basic_config(), invalid_config(), sample_inputs(), mock_distilbert_output(), test_joint_model_output_initialization_empty(self), test_joint_model_output_initialization_with_tensors(self), test_joint_model_output_inheritance(self), test_joint_model_output_dict_access(self), test_initialization_with_valid_config(self, basic_config), test_initialization_with_invalid_config(self, invalid_config), test_initialization_with_missing_intent_labels(self), test_initialization_with_missing_slot_labels(self), test_initialization_with_custom_dropout(self), test_initialization_with_default_dropout(self), test_linear_layer_dimensions(self, basic_config), test_forward_inference_mode(self, mock_distilbert_class, basic_config, sample_inputs, mock_distilbert_output), test_forward_training_mode(self, mock_distilbert_class, basic_config, sample_inputs, mock_distilbert_output), test_forward_return_dict_false(self, mock_distilbert_class, basic_config, sample_inputs, mock_distilbert_output), test_forward_with_all_optional_args(self, mock_distilbert_class, basic_config, sample_inputs, mock_distilbert_output), test_loss_calculation_normal_case(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_loss_with_padded_tokens(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_loss_with_all_padded_tokens(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_loss_with_zero_slot_labels(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_no_loss_when_only_intent_labels_provided(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_no_loss_when_only_slot_labels_provided(self, mock_distilbert_class, basic_config, mock_distilbert_output), test_forward_with_empty_batch(self, mock_distilbert_class, basic_config), test_forward_with_single_token_sequence(self, mock_distilbert_class, basic_config), test_forward_with_very_long_sequence(self, mock_distilbert_class, basic_config), test_model_device_consistency(self, basic_config), test_model_parameters_require_grad(self, basic_config), test_model_eval_mode(self, basic_config), test_model_train_mode(self, basic_config), test_complete_training_step(self, mock_distilbert_class, basic_config), test_batch_size_consistency(self, mock_distilbert_class, basic_config), test_sequence_length_consistency(self, mock_distilbert_class, seq_len, basic_config), test_different_label_counts(self, num_intent_labels, num_slot_labels), test_model_memory_efficiency(self, basic_config), test_model_reproducibility(self, basic_config), create_mock_distilbert_instance(), test_model_string_representation(self, basic_config), test_model_parameter_count(self, basic_config), test_model_config_access(self, basic_config), test_model_submodule_access(self, basic_config), test_model_inheritance(self, basic_config), test_forward_pass_timing(self, mock_distilbert_class, basic_config), test_memory_usage(self, mock_distilbert_class, basic_config), teardown_module()

**Imports:** gc, torch, unittest.mock, sys, os, time, transformers.modeling_outputs, modules.joint_model, pytest, transformers.utils, transformers

## test_model_training

**Functions:** temp_dir(), sample_dataset_csv(temp_dir), empty_dataset_csv(temp_dir), malformed_dataset_csv(temp_dir), test_fine_tune_model_with_valid_dataset_succeeds(self, mock_trainer, mock_model, mock_config, mock_tokenizer, mock_load_dataset, sample_dataset_csv, temp_dir), test_fine_tune_model_creates_correct_label_mappings(self, mock_trainer, mock_model, mock_config, mock_tokenizer, mock_load_dataset, sample_dataset_csv, temp_dir), test_fine_tune_model_with_empty_dataset_handles_gracefully(self, mock_load_dataset, empty_dataset_csv, temp_dir), test_fine_tune_model_with_single_label_dataset(self, mock_config, mock_tokenizer, mock_load_dataset, temp_dir), test_fine_tune_model_with_malformed_entities_json(self, mock_load_dataset, temp_dir), test_fine_tune_model_with_nonexistent_dataset_raises_error(self, temp_dir), test_fine_tune_model_with_missing_required_columns_raises_error(self, mock_load_dataset, malformed_dataset_csv, temp_dir), test_fine_tune_model_with_invalid_model_save_path_raises_error(self, mock_load_dataset, sample_dataset_csv), test_fine_tune_model_with_trainer_failure_raises_error(self, mock_model, mock_config, mock_tokenizer, mock_load_dataset, sample_dataset_csv, temp_dir), test_fine_tune_model_with_unsupported_dataset_type_raises_error(self, mock_load_dataset, sample_dataset_csv, temp_dir), test_fine_tune_model_cpu_only_training(self, mock_trainer, mock_model, mock_config, mock_tokenizer, mock_load_dataset, mock_cuda, sample_dataset_csv, temp_dir), test_fine_tune_model_creates_directory_structure(self, mock_trainer, mock_model, mock_config, mock_tokenizer, mock_load_dataset, sample_dataset_csv, temp_dir), test_process_data_for_joint_model_with_valid_entities(self), test_entity_to_slot_label_conversion(self, entity_json, expected_slots), test_main_function_with_valid_arguments(self, mock_isfile, mock_fine_tune), test_main_function_with_nonexistent_dataset(self, mock_isfile), test_normalize_text_integration(self), test_training_arguments_configuration(self), test_model_config_setup(self), test_all_public_functions_have_tests(self), test_training_completes_within_time_limit(self), test_memory_usage_during_training(self)

**Imports:** tempfile, unittest.mock, sys, datasets, pandas, modules.model_training, pytest, shutil, inspect, os

## test_reminder_utils

**Functions:** mock_datetime(), sample_entities(), test_parse_time_basic_formats(self, mock_datetime), test_parse_time_relative_formats(self, mock_datetime), test_parse_time_default_values(self, mock_datetime), test_parse_time_invalid_inputs(self), test_parse_time_edge_cases(self, mock_datetime), test_parse_date_basic_references(self, mock_dt, mock_datetime), test_parse_date_week_references(self, mock_dt, mock_datetime), test_parse_date_day_of_week(self, mock_dt, mock_datetime), test_parse_date_all_weekdays(self, mock_dt, mock_datetime, day_name, expected_offset), test_parse_date_invalid_inputs(self), test_parse_date_case_insensitive(self, mock_dt, mock_datetime), test_parse_reminder_with_entities(self, mock_dt, mock_datetime), test_parse_reminder_with_date_reference_and_time(self, mock_dt, mock_datetime), test_parse_reminder_fallback_to_regex(self, mock_dt, mock_datetime), test_parse_reminder_regex_patterns(self, mock_dt, mock_datetime), test_parse_reminder_task_extraction(self, mock_dt, mock_datetime, text, expected_task), test_parse_reminder_invalid_inputs(self), test_parse_reminder_edge_cases(self, mock_dt, mock_datetime), test_parse_reminder_relative_time_units(self, mock_dt, mock_datetime), test_parse_list_reminder_with_entities(self, mock_dt, mock_datetime), test_parse_list_reminder_basic_dates(self, mock_dt, mock_datetime), test_parse_list_reminder_relative_dates(self, mock_dt, mock_datetime), test_parse_list_reminder_weekdays(self, mock_dt, mock_datetime), test_parse_list_reminder_next_this_modifiers(self, mock_dt, mock_datetime), test_parse_list_reminder_month_dates(self, mock_dt, mock_datetime), test_parse_list_reminder_formatted_dates(self, mock_dt, mock_datetime), test_parse_list_reminder_all_months(self, mock_dt, mock_datetime, month_name, month_num), test_parse_list_reminder_invalid_inputs(self), test_parse_list_reminder_case_insensitive(self, mock_dt, mock_datetime), test_parse_list_reminder_edge_cases(self, mock_dt, mock_datetime), test_reminder_parsing_integration(self, mock_dt, mock_datetime), test_entity_vs_regex_consistency(self, mock_dt, mock_datetime), test_performance_large_text_input(self), test_memory_usage_stress_test(self), test_concurrent_parsing_safety(self, mock_dt, mock_datetime), worker(idx), test_malformed_input_handling(self), test_unicode_and_special_characters(self), test_timezone_awareness(self), reset_datetime_mock(), test_module_imports(), test_function_signatures()

**Imports:** gc, unittest.mock, threading, sys, time, pytest, datetime, inspect, modules.reminder_utils, os

## test_weather_service

**Functions:** mock_api_key(), mock_aiohttp_get(), mock_weather_success_payload(), mock_ip_geo_success_payload(), mock_ip_geo_failure_payload()

**Imports:** unittest.mock, sys, pytest, aiohttp, modules.weather_service, json, os
