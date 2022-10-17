import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self, config, prefix_projection=True):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.prefix_len, config.hidden_size)
            self.trans1 = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, 
                                config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prefix_len, 
                                                config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans1(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def get_prompt(cls, prefix_token_ids, batch_size, device):
    # prefix_token_ids: List
    prefix_token_ids = torch.Tensor(prefix_token_ids).long()
    prefix_tokens = prefix_token_ids.unsqueeze(0).expand(batch_size, -1).to(device)
    past_key_values = cls.prefix_encoder(prefix_tokens)
    # bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(
        batch_size,
        prefix_token_ids.size(0),
        cls.n_layer * 2, 
        cls.n_head,
        cls.n_embd
    )
    past_key_values = cls.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # 12-length tuple with each: (2, bz, head, prefix_len, hid_emb)
    return past_key_values


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def get_delta(cls, encoder, enc_template, enc_bs, device, prefix_ids=None):    
    
    max_sent_len = 128

    # prefix: past_key_value
    if cls.model_args.use_prefix:
        assert prefix_ids is not None
        past_key_values = get_prompt(cls, prefix_ids, max_sent_len, device)
    else:
        past_key_values = None
        prefix_ids = list()

    # Prepare input ids
    d_input_ids = torch.Tensor(enc_template).repeat(max_sent_len, 1).to(device).long()

    # Prepare position ids
    template_len = d_input_ids.shape[1]
    if cls.model_args.use_prefix:
        offset_len = cls.prefix_len
    else:
        offset_len = 0
    d_position_ids = torch.arange(offset_len, template_len+offset_len).to(device).unsqueeze(0).repeat(max_sent_len, 1).long()
    d_position_ids[:, len(enc_bs):] += torch.arange(max_sent_len).to(device).unsqueeze(-1)
    
    outputs = encoder(input_ids=d_input_ids,
                      position_ids=d_position_ids,
                      past_key_values=past_key_values,
                      output_hidden_states=False, 
                      return_dict=True)
    last_hidden = outputs.last_hidden_state
    m_mask = d_input_ids == cls.mask_token_id
    delta = last_hidden[m_mask]
    template_len = d_input_ids.shape[1]
    return delta, template_len


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    if cls.model_args.sup_label_num > 0:
        cls.sup_classifier = nn.Linear(config.hidden_size*3, cls.model_args.sup_label_num)

    if cls.model_args.use_prefix:
        cls.prefix_encoder = PrefixEncoder(config)

    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    sup_input_ids=None,
    sup_attention_mask=None,
    stage=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    if stage == "1":
        batch_size, num_sent = sup_input_ids.size(0), sup_input_ids.size(1)
        input_ids = sup_input_ids.view(-1, sup_input_ids.size(-1))
        attention_mask = sup_attention_mask.view(-1, sup_attention_mask.size(-1))
        sup_batch_size, sup_num_sent = 0, 0
    else:
        assert stage == "2"
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = None
            # token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

        if cls.model_args.use_aux_loss and sup_input_ids is not None:
            assert sup_input_ids is not None

            sup_batch_size, sup_num_sent = sup_input_ids.size(0), sup_input_ids.size(1)
            sup_input_ids = sup_input_ids.view(-1, sup_input_ids.size(-1))
            sup_attention_mask = sup_attention_mask.view(-1, sup_attention_mask.size(-1))


            if sup_input_ids.size(-1) != input_ids.size(-1):
                length_diff = sup_input_ids.size(-1) - input_ids.size(-1)
                padding = torch.zeros(batch_size*num_sent, length_diff).long().to(cls.device)
                input_ids = torch.cat([input_ids, padding], dim=1)
                attention_mask = torch.cat([attention_mask, padding], dim=1)
            assert sup_input_ids.shape[-1] == input_ids.shape[-1]


            input_ids = torch.cat([input_ids, sup_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, sup_attention_mask], dim=0)
        else:
            sup_batch_size, sup_num_sent = 0, 0


    if cls.model_args.use_prefix:
        prefix_ids = list(range(cls.prefix_len))
        past_key_values = get_prompt(cls, prefix_ids, batch_size*num_sent+sup_batch_size*sup_num_sent, cls.device)

        #-------------------------------------------
        # Prepare attention mask for past_key_value
        #-------------------------------------------
        if stage == "1":
            pones = torch.ones(batch_size, cls.prefix_len//2).to(cls.device)
            pzros = torch.zeros(batch_size, cls.prefix_len//2).to(cls.device)
            prefix_att_mask_left  = torch.cat([pones, pzros, pzros], dim=1).reshape(-1, cls.prefix_len//2)
            prefix_att_mask_right = torch.cat([pzros, pones, pones], dim=1).reshape(-1, cls.prefix_len//2)
        else:
            assert stage == "2"
            pones = torch.ones(batch_size, cls.prefix_len//2).to(cls.device)
            pzros = torch.zeros(batch_size, cls.prefix_len//2).to(cls.device)
            if num_sent == 2:
                prefix_att_mask_left  = torch.cat([pones, pzros], dim=1).reshape(-1, cls.prefix_len//2)
                prefix_att_mask_right = torch.cat([pzros, pones], dim=1).reshape(-1, cls.prefix_len//2)
            elif num_sent == 3:
                prefix_att_mask_left  = torch.cat([pones, pzros, pzros], dim=1).reshape(-1, cls.prefix_len//2)
                prefix_att_mask_right = torch.cat([pzros, pones, pones], dim=1).reshape(-1, cls.prefix_len//2)
            else:
                raise NotImplementedError

            if cls.model_args.use_aux_loss and sup_input_ids is not None:
                assert sup_input_ids is not None
                sup_pones =  torch.ones(sup_batch_size, cls.prefix_len//2).to(cls.device)
                sup_pzros = torch.zeros(sup_batch_size, cls.prefix_len//2).to(cls.device)
                prefix_sup_att_mask_left = torch.cat([sup_pones, sup_pzros, sup_pzros], dim=1).reshape(-1, cls.prefix_len//2)
                prefix_sup_att_mask_right = torch.cat([sup_pzros, sup_pones, sup_pones], dim=1).reshape(-1, cls.prefix_len//2)

                prefix_att_mask_left = torch.cat([prefix_att_mask_left, prefix_sup_att_mask_left], dim=0)
                prefix_att_mask_right = torch.cat([prefix_att_mask_right, prefix_sup_att_mask_right], dim=0)

        attention_mask = torch.cat((prefix_att_mask_left, prefix_att_mask_right, attention_mask), dim=1).long()
        #-------------------------------------------
    else:
        past_key_values = None


    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        past_key_values=past_key_values,
        output_hidden_states=False,
        return_dict=True,
    )

    # Pooling
    if cls.model_args.apply_prompt:
        pooler_output = outputs.last_hidden_state[input_ids == cls.mask_token_id]
    else:
        pooler_output = outputs.last_hidden_state[:, 0]

    # restore shape: (bs, num_sent, hid_emb)
    if stage == "1":
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))
        attention_mask = attention_mask.view(batch_size, num_sent, -1)
    else:
        assert stage == "2"
        if sup_input_ids is not None:
            assert cls.model_args.use_aux_loss
            total_unsup_sent = batch_size * num_sent
            sup_pooler_output = pooler_output[total_unsup_sent:, :].view(sup_batch_size, sup_num_sent, -1) # (bs, num_sent=3, hidden)
            sup_attention_mask= attention_mask[total_unsup_sent:, :].view(sup_batch_size, sup_num_sent, -1) # (bs, num_sent=3, hidden)
            pooler_output = pooler_output[:total_unsup_sent, :].view(batch_size, num_sent, -1) # (bs, num_sent=2, hidden)
            attention_mask = attention_mask[:total_unsup_sent, :].view(batch_size, num_sent, -1) # (bs, num_sent=2, hidden)
        else:
            pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
            attention_mask = attention_mask.view(batch_size, num_sent, -1)

    # Template denoising (for now assume only for CL)
    if stage == "2" and cls.model_args.apply_template_delta_train:
        if cls.model_args.use_prefix:
            prefix_id_for_one = list(range(cls.prefix_len//2))
            prefix_id_for_two = list(range(cls.prefix_len//2, cls.prefix_len))
        else:
            prefix_id_for_one, prefix_id_for_two = None, None
        delta, template_len =   get_delta(cls, encoder, cls.enc_template, cls.bs, cls.device, prefix_ids=prefix_id_for_one)
        delta2, template_len2 = get_delta(cls, encoder, cls.enc_template, cls.bs, cls.device, prefix_ids=prefix_id_for_two)
        
        # attention_mask = attention_mask.view(batch_size, num_sent, -1)
        blen = attention_mask.sum(-1) - template_len
        pooler_output[:, 0, :] -= delta[blen[:, 0]]
        blen2 = attention_mask.sum(-1) - template_len2
        pooler_output[:, 1, :] -= delta2[blen2[:, 1]]
        if num_sent == 3:
            pooler_output[:, 2, :] -= delta2[blen2[:, 2]]

    if cls.model_args.mlp_train:
        pooler_output = cls.mlp(pooler_output)
        if stage == "2" and cls.model_args.use_aux_loss and sup_input_ids is not None:
            sup_pooler_output = cls.mlp(sup_pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    if num_sent == 3:
        z3 = pooler_output[:, 2]
    if stage == "2" and cls.model_args.use_aux_loss and sup_input_ids is not None:
        sup_z1, sup_z2, sup_z3 = sup_pooler_output[:,0], sup_pooler_output[:,1], sup_pooler_output[:,2]

    # Gather all embeddings if using distributed training (for now assume only for CL)
    if stage == "2" and dist.is_initialized() and cls.training:
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)


    if stage == "1":
        assert num_sent == 3       
        positive_feat = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=1)
        negative_feat = torch.cat([z1, z3, torch.abs(z1 - z3)], dim=1)
        combined_feat = torch.cat([positive_feat, negative_feat], dim=0)
        positive_labels = torch.ones(positive_feat.size(0)).long().to(cls.device)
        negative_labels = torch.zeros(negative_feat.size(0)).long().to(cls.device)
        nli_labels = torch.cat([positive_labels, negative_labels])

        nli_prediction = cls.sup_classifier(combined_feat)
        nli_loss_fn = nn.CrossEntropyLoss()
        loss = nli_loss_fn(nli_prediction, nli_labels)
    else:
        assert stage == "2"
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(cos_sim, labels)
        if cls.model_args.use_aux_loss and sup_input_ids is not None:         
            positive_feat = torch.cat([sup_z1, sup_z2, torch.abs(sup_z1 - sup_z2)], dim=1)
            negative_feat = torch.cat([sup_z1, sup_z3, torch.abs(sup_z1 - sup_z3)], dim=1)
            combined_feat = torch.cat([positive_feat, negative_feat], dim=0)
            positive_labels = torch.ones(positive_feat.size(0)).long().to(cls.device)
            negative_labels = torch.zeros(negative_feat.size(0)).long().to(cls.device)
            nli_labels = torch.cat([positive_labels, negative_labels])
            
            nli_prediction = cls.sup_classifier(combined_feat)
            nli_loss_fn = nn.CrossEntropyLoss()
            nli_loss = nli_loss_fn(nli_prediction, nli_labels)

            loss += cls.model_args.aux_weight * nli_loss

    return SequenceClassifierOutput(
        loss=loss,
        # logits=cos_sim,
        logits=None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.shape[0]

    if cls.model_args.use_prefix:
        prefix_ids = list(range(cls.prefix_len))
        past_key_values = get_prompt(cls, prefix_ids, batch_size, input_ids.device)
        prefix_attention_mask_ones = torch.ones(batch_size, cls.prefix_len).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask_ones, attention_mask), dim=1).long()
    else:
        past_key_values = None

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        past_key_values=past_key_values,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.apply_prompt:
        pooler_output = outputs.last_hidden_state[input_ids == cls.mask_token_id]
    else:
        pooler_output = outputs.last_hidden_state[:, 0]
    
    if cls.model_args.apply_template_delta_infer:
        if cls.model_args.use_prefix:
            prefix_id_for_eval = list(range(cls.prefix_len))
        else:
            prefix_id_for_eval = None
        delta, template_len = get_delta(cls, encoder, cls.enc_template, cls.bs, input_ids.device, prefix_ids=prefix_id_for_eval)
        blen = attention_mask.sum(-1) - template_len
        pooler_output -= delta[blen]

    if cls.model_args.mlp_eval:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        if self.model_args.apply_prompt:
            self.mask_token_id = self.model_args.mask_token_id
            self.bs, self.es = self.model_args.enc_bs, self.model_args.enc_es 
            self.enc_template =  self.model_args.enc_template

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        if self.model_args.use_prefix:
            config.prefix_len = self.model_args.prefix_len
            self.prefix_len = config.prefix_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
        else:
            self.prefix_len = 0

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        sup_input_ids=None,
        sup_attention_mask=None,
        stage="2",
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                sup_input_ids=sup_input_ids,
                sup_attention_mask=sup_attention_mask,
                stage=stage
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        if self.model_args.apply_prompt:
            self.mask_token_id = self.model_args.mask_token_id
            self.bs, self.es = self.model_args.enc_bs, self.model_args.enc_es 
            self.enc_template =  self.model_args.enc_template

        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.prefix_len = 0
        if self.model_args.use_prefix:
            config.prefix_len = self.model_args.prefix_len
            self.prefix_len = config.prefix_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        sup_input_ids=None,
        sup_attention_mask=None,
        stage="2"
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                sup_input_ids=sup_input_ids,
                sup_attention_mask=sup_attention_mask,
                stage=stage,
            )
