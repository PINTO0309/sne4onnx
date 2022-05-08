#! /usr/bin/env python

import sys
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
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
    input_op_names: List[str],
    output_op_names: List[str],
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:

    """
    Parameters
    ----------
    input_op_names: List[str]
        List of OP names to specify for the input layer of the model.\n\
        e.g. ['aaa','bbb','ccc']

    output_op_names: List[str]
        List of OP names to specify for the output layer of the model.\n\
        e.g. ['ddd','eee','fff']

    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If not specified, .onnx is not output.\n\
        Default: ''

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

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

    if not input_op_names:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One or more input_op_names must be specified.'
        )
        sys.exit(1)

    if not output_op_names:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One or more output_op_names must be specified.'
        )
        sys.exit(1)

    # Load
    graph = None
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Extraction of input OP and output OP
    graph_node_inputs = [
        graph_nodes \
            for graph_nodes in graph.nodes \
                for graph_nodes_input in graph_nodes.inputs \
                    if graph_nodes_input.name in input_op_names
    ]
    graph_node_outputs = [
        graph_nodes \
            for graph_nodes in graph.nodes \
                for graph_nodes_output in graph_nodes.outputs \
                    if graph_nodes_output.name in output_op_names
    ]

    # Init graph INPUT/OUTPUT
    graph.inputs.clear()
    graph.outputs.clear()

    # Update graph INPUT/OUTPUT
    graph.inputs = [
        graph_node_input \
            for graph_node in graph_node_inputs \
                for graph_node_input in graph_node.inputs \
                    if graph_node_input.shape
    ]
    graph.outputs = [
        graph_node_output \
            for graph_node in graph_node_outputs \
                for graph_node_output in graph_node.outputs
    ]

    # Cleanup
    graph.cleanup().toposort()

    # Shape Estimation
    extracted_graph = None
    try:
        extracted_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except Exception as e:
        extracted_graph = gs.export_onnx(graph)
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
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
        nargs='+',
        required=True,
        help="\
            List of OP names to specify for the input layer of the model. \
            e.g. --input_op_names aaa bbb ccc"
    )
    parser.add_argument(
        '--output_op_names',
        type=str,
        nargs='+',
        required=True,
        help="\
            List of OP names to specify for the output layer of the model. \
            e.g. --output_op_names ddd eee fff"
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='extracted.onnx',
        help='Output onnx file path. If not specified, extracted.onnx is output.'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    input_op_names = args.input_op_names
    output_op_names = args.output_op_names
    output_onnx_file_path = args.output_onnx_file_path
    non_verbose = args.non_verbose

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)

    # Model extraction
    extracted_graph = extraction(
        input_onnx_file_path=None,
        input_op_names=input_op_names,
        output_op_names=output_op_names,
        onnx_graph=onnx_graph,
        output_onnx_file_path=output_onnx_file_path,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()
