#! /usr/bin/env python

from argparse import ArgumentParser
import onnx
from typing import List

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
    output_onnx_file_path: str,
    input_op_names: List[str],
    output_op_names: List[str],
):

    """
    Parameters
    ----------
    input_onnx_file_path: str
        Input onnx file path.

    output_onnx_file_path: str
        Output onnx file path.

    input_op_names: List[str]
        List of OP names to specify for the input layer of the model.\n\
        Specify the name of the OP, separated by commas.\n\
        e.g. ['aaa','bbb','ccc']

    output_op_names: List[str]
        List of OP names to specify for the output layer of the model.\n\
        Specify the name of the OP, separated by commas.\n\
        e.g. ['ddd','eee','fff']
    """

    onnx.utils.extract_model(
        input_onnx_file_path,
        output_onnx_file_path,
        input_op_names,
        output_op_names
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
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
    args = parser.parse_args()

    input_op_names = args.input_op_names.strip(' ,').replace(' ','').split(',')
    output_op_names = args.output_op_names.strip(' ,').replace(' ','').split(',')

    # Model extraction
    extraction(
        input_onnx_file_path=args.input_onnx_file_path,
        output_onnx_file_path=args.output_onnx_file_path,
        input_op_names=input_op_names,
        output_op_names=output_op_names,
    )

if __name__ == '__main__':
    main()