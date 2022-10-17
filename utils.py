def write_eval_args(model_args, train_args):
    '''
    To write args json for evaluation based training args
    '''
    eval_args = dict()
                                                    
    eval_args['model_name_or_path'] = train_args.output_dir
    eval_args['mode'] = 'test'
    eval_args['task_set'] = 'sts'

    eval_args['apply_prompt'] = model_args.apply_prompt
    if model_args.apply_prompt:
        eval_args['prompt_list'] = [model_args.prompt_template_id]
        eval_args['eval_template'] = model_args.eval_template
        if model_args.apply_template_delta_infer:
            eval_args['apply_template_delta'] = True
        else:
            eval_args['apply_template_delta'] = False
        
        eval_args['pooler'] = 'avg'
        if model_args.mlp_only_train:
            eval_args['apply_mlp_pooler'] = False
        else:
            eval_args['apply_mlp_pooler'] = True

        if model_args.apply_prompt_token: 
            eval_args['apply_prompt_token'] = True
        else: 
            eval_args['apply_prompt_token'] = False

    elif model_args.mlp_only_train:
        eval_args['pooler'] = 'cls_before_pooler'
    else:
        eval_args['pooler'] = 'cls'
    return eval_args

TEMPLATES = {
    "00": '*cls*_This_sentence_of_"*sent0*"_means*mask*.*sep+*',
    "0": '*cls*_This_sentence_:_"*sent0*"_means*mask*.*sep+*',
    "1": '*cls*_This_sentence_:_"_*sent0*_"_means_"_*mask*_".*sep+*',
    "2": '*cls*_"_*mask*_"_means_"_*sent0*_"_._*sep+*',
    "3": '*cls*_This_token_"_*mask*_"_means_"_*sent0*_"_._*sep+*',
    "4": '*cls*_The_sentence_:_"_*mask*_"_means_"_*sent0*_"_._*sep+*',
    "5": '*cls*_The_sentence_:_"_*mask*_"_has_the_meaning_of_"_*sent0*_".*sep+*',
    "6": "*cls*_This_sentence_:_'_*sent0*_'_means*mask*.*sep+*",
    "7": "*cls*_The_sentence_:_'_*sent0*_'_means*mask*.*sep+*",
    "8": "*cls*_The_sentence_:_'_*sent0*_'_means_'_*mask*_'.*sep+*",
    "9": "*cls*_The_sentence_:_'*mask*'_means_'_*sent0*_'.*sep+*",
}


def change_templates(template: str) -> str:
    output_string = "*cls*"
    for i, s in enumerate(template[5:-7]):
        if s == "@":
            return template
        current_pos = i + 5
        
        if template[current_pos-1] == "_" and template[current_pos] != "*":
            output_string += "@"
        
        output_string += s
        
        if template[current_pos+1] == "_" and (template[current_pos] != "*" and
                                               template[current_pos] != '"'):
            output_string += "@"

    output_string += "_*sep+*"
    return output_string


def get_encoded_bs_and_es(template, tokenizer):
    templt = template.replace('*mask*', tokenizer.mask_token)\
                       .replace('*sep+*', '').replace('*cls*', '').replace('*sent0*', ' ').split(" ")
    
    bs_tokens = templt[0].replace('_', ' ').strip()
    enc_bs = tokenizer.encode(bs_tokens, add_special_tokens=False)
    
    es_tokens = templt[1].replace('_', ' ').strip()
    enc_es = tokenizer.encode(es_tokens, add_special_tokens=False)

    mask_embedding_sentence_bs = templt[0].replace('_', ' ')
    if 'roberta' in tokenizer.name_or_path:
        # remove empty block
        mask_embedding_sentence_bs = mask_embedding_sentence_bs.strip()
    mask_embedding_sentence_es = templt[1].replace('_', ' ')

    enc_bs = tokenizer.encode(mask_embedding_sentence_bs, add_special_tokens=False)        
    enc_es = tokenizer.encode(mask_embedding_sentence_es, add_special_tokens=False)
    enc_template = tokenizer.encode(mask_embedding_sentence_bs + mask_embedding_sentence_es)

    template_tokens = template.replace('*mask*', tokenizer.mask_token)\
                       .replace('*sep+*', '').replace('*cls*', '').replace('*sent0*', '[sent]').replace('_', ' ')
    return bs_tokens, es_tokens, template_tokens, [tokenizer.cls_token_id] + enc_bs, enc_es + [tokenizer.sep_token_id], enc_template
