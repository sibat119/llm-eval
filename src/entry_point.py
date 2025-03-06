from src import (
    args,
    utils,
    cfg_reader,
    finetune,
    pretrain
)


def main():
    args = args.parse()
    cfg = cfg_reader.primary.load(args.config)
    
    if args.procedure == 'finetune':
        finetune.main(args, cfg)
    elif args.procedure == 'pretrain':
        pretrain.main(args, cfg)
    else:
        raise NotImplementedError(utils.strings.clean_multiline(
            """
            Procedure added to args but case not added to main function in <project root>/lm_toolkit/__init__.py.
            """
        ))
            
