#! /usr/bin/env python

import sys
from argparse import ArgumentParser
import onnx
from onnx.external_data_helper import uses_external_data
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

ONNX_STANDARD_DOMAINS = [
    'ai.onnx',
    'ai.onnx.ml',
    '',
]


def model_uses_external_data(input_onnx_file_path: str) -> bool:
    model = onnx.load(input_onnx_file_path, load_external_data=False)
    def iter_tensors_in_graph(g):
        for t in g.initializer:
            yield t
        for t in g.sparse_initializer:
            yield t
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    yield a.t
                elif a.type == onnx.AttributeProto.TENSORS:
                    for t in a.tensors:
                        yield t
                elif a.type == onnx.AttributeProto.GRAPH:
                    yield from iter_tensors_in_graph(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        yield from iter_tensors_in_graph(sg)
    return any(uses_external_data(t) for t in iter_tensors_in_graph(model.graph))

def extraction(
    input_op_names: List[str],
    output_op_names: List[str],
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    has_external_data: Optional[bool] = False,
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

    has_external_data: Optional[bool]
        Default: False

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
        has_external_data = model_uses_external_data(input_onnx_file_path)
    if not has_external_data:
        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)

    # Acquisition of Node with custom domain
    custom_domain_check_onnx_nodes = []
    custom_domain_check_onnx_nodes = \
        custom_domain_check_onnx_nodes + \
            [
                node for node in onnx_graph.graph.node \
                    if node.domain not in ONNX_STANDARD_DOMAINS
            ]

    # domain, ir_version
    domain: str = onnx_graph.domain
    ir_version: int = onnx_graph.ir_version
    meta_data = {'domain': domain, 'ir_version': ir_version}
    metadata_props = None
    if hasattr(onnx_graph, 'metadata_props'):
        metadata_props = onnx_graph.metadata_props

    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Check if Graph contains a custom domain (custom module)
    contains_custom_domain = len(
        [
            domain \
                for domain in graph.import_domains \
                    if domain.domain not in ONNX_STANDARD_DOMAINS
        ]
    ) > 0

    # Extraction of input OP and output OP
    graph_node_inputs = [
        graph_node \
            for graph_node in graph.nodes \
                for graph_nodes_input in graph_node.inputs \
                    if graph_nodes_input.name in input_op_names
    ]
    graph_node_outputs = [
        graph_node \
            for graph_node in graph.nodes \
                for graph_nodes_output in graph_node.outputs \
                    if graph_nodes_output.name in output_op_names
    ]

    # Init graph INPUT/OUTPUT
    graph.inputs.clear()
    graph.outputs.clear()

    # Update graph INPUT/OUTPUT
    input_tmp = []
    for graph_node in graph_node_inputs:
        for graph_node_input in graph_node.inputs:
            if graph_node_input.shape \
                and graph_node_input not in [i for i in input_tmp] \
                and hasattr(graph_node_input, 'name') \
                and graph_node_input.name in [i for i in input_op_names]:
                input_tmp.append(graph_node_input)
    graph.inputs = input_tmp

    graph.outputs = [
        graph_node_output \
            for graph_node in graph_node_outputs \
                for graph_node_output in graph_node.outputs
    ]

    # Cleanup
    graph.cleanup(remove_unused_graph_inputs=True).toposort()

    # Shape Estimation
    extracted_graph = None
    try:
        exported_onnx_graph = gs.export_onnx(graph, do_type_check=False, **meta_data)
        if metadata_props is not None:
            exported_onnx_graph.metadata_props.extend(metadata_props)
        extracted_graph = onnx.shape_inference.infer_shapes(exported_onnx_graph)
    except Exception as e:
        extracted_graph = gs.export_onnx(graph, do_type_check=False, **meta_data)
        if metadata_props is not None:
            extracted_graph.metadata_props.extend(metadata_props)
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )

    ## 5. Restore a node's custom domain
    if contains_custom_domain:
        extracted_graph_nodes = extracted_graph.graph.node
        for extracted_graph_node in extracted_graph_nodes:
            for custom_domain_check_onnx_node in custom_domain_check_onnx_nodes:
                if extracted_graph_node.name == custom_domain_check_onnx_node.name:
                    extracted_graph_node.domain = custom_domain_check_onnx_node.domain

    # Save
    if output_onnx_file_path:
        onnx.save(extracted_graph, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    return extracted_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '-ion',
        '--input_op_names',
        type=str,
        nargs='+',
        required=True,
        help="\
            List of OP names to specify for the input layer of the model. \
            e.g. --input_op_names aaa bbb ccc"
    )
    parser.add_argument(
        '-oon',
        '--output_op_names',
        type=str,
        nargs='+',
        required=True,
        help="\
            List of OP names to specify for the output layer of the model. \
            e.g. --output_op_names ddd eee fff"
    )
    parser.add_argument(
        '-of',
        '--output_onnx_file_path',
        type=str,
        default='extracted.onnx',
        help='Output onnx file path. If not specified, extracted.onnx is output.'
    )
    parser.add_argument(
        '-n',
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
    # External Data Check
    has_external_data = model_uses_external_data(input_onnx_file_path)

    # Model extraction
    extracted_graph = extraction(
        input_onnx_file_path=None,
        input_op_names=input_op_names,
        output_op_names=output_op_names,
        onnx_graph=onnx_graph,
        output_onnx_file_path=output_onnx_file_path,
        has_external_data=has_external_data,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()
