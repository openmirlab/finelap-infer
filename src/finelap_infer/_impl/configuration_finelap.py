from transformers import PretrainedConfig

from .configuration_eat import EATConfig


class FineLAPConfig(PretrainedConfig):
    model_type = "finelap"

    def __init__(
        self,
        embed_size=1024,
        temp_global=0.1,
        b_global=-10.0,
        temp_local=0.1,
        b_local=-10.0,
        local_audio_proj_type="transformer",
        normalize_dense_audio_embeds=True,
        unify_audio_proj=False,
        text_encoder_name="roberta-base",
        audio_config=None,
        **kwargs
    ):
        self.embed_size = embed_size
        self.temp_global = temp_global
        self.b_global = b_global
        self.temp_local = temp_local
        self.b_local = b_local
        self.local_audio_proj_type = local_audio_proj_type
        self.normalize_dense_audio_embeds = normalize_dense_audio_embeds
        self.unify_audio_proj = unify_audio_proj
        self.text_encoder_name = text_encoder_name

        if isinstance(audio_config, dict):
            self.audio_config = EATConfig(**audio_config)
        elif isinstance(audio_config, EATConfig):
            self.audio_config = audio_config
        else:
            self.audio_config = EATConfig()

        super().__init__(**kwargs)
