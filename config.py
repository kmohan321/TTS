config = {
    "audio_encoder": {
        "strides": [6, 4, 4, 2], 
        "input_channels": 1,
        "hidden_channels": 64,
        "latent_channels": 512,
        "kernel": 7,
        "num_codebook": 4,
        "codebook_size": 2048,
        "codebook_dim": 512
    },

    "transformer": {
        "num_blocks": 2,
        "hidden_dims": 256,
        "num_heads_q": 8,
        "num_heads_kv": 4,
        "seq_length": 1000,
        "ffn_multiplier": 4,
        "vocab_size": 52000,
        "eps": 1e-5,
        "head_dim": 256//8,
        "num_cd_vectors": 2048,
        "num_codebooks": 4,
        "max_audio_seq_length": 300,
        "max_cond_seq_length": 400
    },

    "training": {
        "epochs": 100,
        "ar_lr": 1e-4,
        "enc_lr": 1e-4
    }
}
