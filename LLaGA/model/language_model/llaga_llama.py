#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from utils.constants import IGNORE_INDEX


class LlagaConfig(LlamaConfig):
    model_type = "llaga"


class LlagaLlamaModel(LlagaMetaModel, LlamaModel):
    config_class = LlagaConfig

    def __init__(self, config: LlamaConfig):
        super(LlagaLlamaModel, self).__init__(config)


class LlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlagaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _prune_tokens(self, inputs_embeds, attention_mask=None, labels=None, prune_token_positions=None):
        if inputs_embeds is None or prune_token_positions is None:
            return inputs_embeds, attention_mask, labels

        if isinstance(prune_token_positions, torch.Tensor):
            prune_positions = prune_token_positions.to(device=inputs_embeds.device, dtype=torch.long).reshape(-1)
        else:
            prune_positions = torch.tensor(prune_token_positions, device=inputs_embeds.device, dtype=torch.long).reshape(-1)

        if prune_positions.numel() == 0:
            return inputs_embeds, attention_mask, labels

        seq_len = inputs_embeds.shape[1]
        prune_positions = prune_positions[(prune_positions >= 0) & (prune_positions < seq_len)].unique(sorted=True)
        if prune_positions.numel() == 0 or prune_positions.numel() >= seq_len:
            return inputs_embeds, attention_mask, labels

        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=inputs_embeds.device)
        keep_mask[prune_positions] = False

        inputs_embeds = inputs_embeds[:, keep_mask, :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, keep_mask]
        if labels is not None:
            labels = labels[:, keep_mask]
        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        prune_token_positions: Optional[torch.LongTensor] = None,
        reposition_perm: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Each generation call
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        inputs_embeds, attention_mask, labels = self._prune_tokens(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            prune_token_positions=prune_token_positions,
        )
        if reposition_perm is not None and inputs_embeds is not None:
            perm = reposition_perm.to(inputs_embeds.device).long().reshape(-1)
            inputs_embeds = inputs_embeds[:, perm, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, perm]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
                "prune_token_positions": kwargs.get("prune_token_positions", None) if past_key_values is None else None,
                "reposition_perm": kwargs.get("reposition_perm", None) if past_key_values is None else None,
            }
        )
        return model_inputs

AutoConfig.register("llaga", LlagaConfig)
AutoModelForCausalLM.register(LlagaConfig, LlagaLlamaForCausalLM)
