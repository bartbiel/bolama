# bolama
<h2>Zero memory</h2>
<br>Activated om my env .\myenv\Scripts\Activate
<h3>Dependencies</h3>
<br> pip install torch transformers
<br> pip install dotenv 
<br>pip install fastapi uvicorn sentencepiece accelerate
<br>pip install protobuf
<h2>Run chat on CLI</h2>
<br>curl.exe -X POST "http://localhost:8000/chat?user_input=Hello!" -H "Content-Type: application/json"

<h2> Some example properties from the run</h2>
<br>Using model from: D:\mistral_models\7B-Instruct-v0.3      
<br>model path for webChat= D:\mistral_models\7B-Instruct-v0.3

<br>=== Core Props ===
<br>Tokenizer class: LlamaTokenizerFast
<br>Vocab size: 32768
<br>Model max length: 1000000000000000019884624838656
<br>Padding side: left
<br>
<br>=== Special Tokens ===
<br>CLS token: None (ID: None)
<br>SEP token: None (ID: None)
<br>PAD token: None (ID: None)
<br>UNK token:  (ID: 0)
<br>BOS token:  (ID: 1)
<br>EOS token:  (ID: 2)
<br>
<br>=== Tokenizer Config ===
<br>no config detected
<br>
<br>=== Tokenization Example ===
<br>Text: 'Hello, how are you?'
<br>Tokens: ['▁Hello', ',', '▁how', '▁are', '▁you', '?']
<br>Token IDs: [1, 23325, 29493, 1678, 1228, 1136, 29572]
<br>
<br>=== Advanced Properties ===
<br>Added tokens: 771
<br>Is fast tokenizer? True
<br>No language deetected
<br>Loading checkpoint shards: 100%|
<br>
<br>=== Architecture ===
<br>Parameters: 7,248,023,552
<br>Layers: 32
<br>Hidden size: 4096
<br>Attention heads: 32
<br>Vocab size: 32768
<br>
<br>=== Config Summary JSON===
<br>
        vocab_size: 32768<br>
        max_position_embeddings: 32768<br>
        hidden_size: 4096<br>
        intermediate_size: 14336<br>
        num_hidden_layers: 32<br>
        num_attention_heads: 32<br>
        sliding_window: None<br>
        head_dim: 128<br>
        num_key_value_heads: 8<br>
        hidden_act: silu<br>
        initializer_range: 0.02<br>
        rms_norm_eps: 1e-05<br>
        use_cache: True<br>
        rope_theta: 1000000.0<br>
        attention_dropout: 0.0<br>
        return_dict: True<br>
        output_hidden_states: False<br>
        output_attentions: False<br>
        torchscript: False<br>
        torch_dtype: float16<br>
        use_bfloat16: False<br>
        tf_legacy_loss: False<br>
        tie_word_embeddings: False<br>
        chunk_size_feed_forward: 0<br>
        is_encoder_decoder: False<br>
        is_decoder: False<br>
        cross_attention_hidden_size: None<br>
        add_cross_attention: False<br>
        tie_encoder_decoder: False<br>
        max_length: 20<br>
        min_length: 0<br>
        do_sample: False<br>
        early_stopping: False<br>
        num_beams: 1<br>
        num_beam_groups: 1<br>
        diversity_penalty: 0.0<br>
        temperature: 1.0<br>
        top_k: 50<br>
        top_p: 1.0<br>
        typical_p: 1.0<br>
        repetition_penalty: 1.0<br>
        length_penalty: 1.0<br>
        no_repeat_ngram_size: 0<br>
        encoder_no_repeat_ngram_size: 0<br>
        bad_words_ids: None<br>
        num_return_sequences: 1<br>
        output_scores: False<br>
        return_dict_in_generate: False<br>
        forced_bos_token_id: None<br>
        forced_eos_token_id: None<br>
        remove_invalid_values: False<br>
        exponential_decay_length_penalty: None<br>
        suppress_tokens: None<br>
        begin_suppress_tokens: None<br>
        finetuning_task: None<br>
        tokenizer_class: None<br>
        prefix: None<br>
        bos_token_id: 1<br>
        pad_token_id: None<br>
        eos_token_id: 2<br>
        sep_token_id: None<br>
        decoder_start_token_id: None<br>
        task_specific_params: None<br>
        problem_type: None<br>
        transformers_version: 4.51.0<br>
        model_type: mistral<br>
<br>
<br>=== Generation Config ===
<br>GenerationConfig {
<br>  "bos_token_id": 1,
<br>  "eos_token_id": 2
<br>}
<br>
<br>Device set to use cpu
<br>=== Pipeline Properties ===
<br>Task: text-generation
<br>Device: cpu
<br>
<br>=== Model ===
<br>Model: MistralForCausalLM
<br>Device: next(chatbot.model.parameters()).device
<br>
<br>=== Tokenizer ===
<br>Tokenizer: LlamaTokenizerFast

<h2>Prompting</h2>
<h3>Dependencies</h3>
<br>pip install langchain
<br>pip install langchain_community
 