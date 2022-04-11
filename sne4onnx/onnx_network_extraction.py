#! /usr/bin/env python

import sys
from argparse import ArgumentParser
import onnx
from typing import Optional, List

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def extraction(
    input_onnx_file_path: str,
    input_op_names: List[str],
    output_op_names: List[str],
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
) -> onnx.ModelProto:

    """
    Parameters
    ----------
    input_onnx_file_path: str
        Input onnx file path.

    input_op_names: List[str]
        List of OP names to specify for the input layer of the model.\n\
        Specify the name of the OP, separated by commas.\n\
        e.g. ['aaa','bbb','ccc']

    output_op_names: List[str]
        List of OP names to specify for the output layer of the model.\n\
        Specify the name of the OP, separated by commas.\n\
        e.g. ['ddd','eee','fff']

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If not specified, .onnx is not output.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    Returns
    -------
    extracted_graph: onnx.ModelProto
        Extracted onnx ModelProto
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Load
    graph = None
    if not onnx_graph:
        graph = onnx.load(input_onnx_file_path)
    else:
        graph = onnx_graph

    # Extract
    extractor = onnx.utils.Extractor(graph)
    extracted_graph = extractor.extract_model(
        input_op_names,
        output_op_names,
    )

    # Save
    if output_onnx_file_path:
        onnx.save(extracted_graph, output_onnx_file_path)

    return extracted_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--input_op_names',
        type=str,
        required=True,
        help="\
            List of OP names to specify for the input layer of the model. \
            Specify the name of the OP, separated by commas. \
            e.g. --input_op_names aaa,bbb,ccc"
    )
    parser.add_argument(
        '--output_op_names',
        type=str,
        required=True,
        help="\
            List of OP names to specify for the output layer of the model. \
            Specify the name of the OP, separated by commas. \
            e.g. --output_op_names ddd,eee,fff"
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='extracted.onnx',
        help='Output onnx file path. If not specified, extracted.onnx is output.'
    )
    args = parser.parse_args()

    input_op_names = args.input_op_names.strip(' ,').replace(' ','').split(',')
    output_op_names = args.output_op_names.strip(' ,').replace(' ','').split(',')

    # Model extraction
    extracted_graph = extraction(
        input_onnx_file_path=args.input_onnx_file_path,
        input_op_names=input_op_names,
        output_op_names=output_op_names,
        output_onnx_file_path=args.output_onnx_file_path,
    )


if __name__ == '__main__':
    main()
