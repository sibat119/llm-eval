import argparse
import argcomplete
from src.utils import files, strings

def parse():
    """
    parse the command line arguments
    """

    ######################### begin arg definitions ######################### 

    # define arguments and parse
    parser = argparse.ArgumentParser() 

    # create subparser for procedures
    subparser = parser.add_subparsers(
        description='decides on which procedure to run',
        required=True,
        dest='procedure',
    )

    # add subparser for playbook generation
    parser_gen = subparser.add_parser(
        'finetune',
        description='finetunes the T5 model'
    )
    parser_gen.add_argument(
        '-c',
        '--config',
        help='config path',
        type=str,
        default="{{PROJECT_ROOT}}/conf/config.yaml",
    )

    parser_gen.add_argument(
        '--debug',
        help='used for testing the create_ansible function',
        action='store_true',
        default=False,
    )
    
    parser_gen.add_argument(
        '--model_name',
        help='Which model to finetune. options: [t5-small, t5-base, t5-large, t5-3b, t5-11b]',
        type=str,
        default="t5-base",
    )
    
    ######################### end arg definitions ######################### 

    # parse
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # perform substitutions
    args.config = strings.replace_slot(
        args.config, 
        { 'PROJECT_ROOT' : files.get_project_root() }
    )

    return args
