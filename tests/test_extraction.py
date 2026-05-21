import os
import sys
import pytest
import onnx

sys.path.insert(0, os.path.dirname(__file__))
from generate_test_model import generate_test_model
from sne4onnx import extraction

TEST_MODEL_PATH = "tests/test_mid_graph_extraction.onnx"


@pytest.fixture(scope="session", autouse=True)
def setup_test_model():
    generate_test_model(TEST_MODEL_PATH)


class TestMidGraphExtraction:
    """Test extracting a sub-graph where the specified input is a mid-graph tensor."""

    def test_input_is_specified_tensor(self):
        """Extracted model input should be 'b', not original input 'a'."""
        model = extraction(
            input_op_names=["b"],
            output_op_names=["d"],
            input_onnx_file_path=TEST_MODEL_PATH,
            non_verbose=True,
        )
        input_names = [inp.name for inp in model.graph.input]
        assert input_names == ["b"]

    def test_upstream_nodes_removed(self):
        """Relu_0 (which produces 'b' from 'a') should not be in the extracted model."""
        model = extraction(
            input_op_names=["b"],
            output_op_names=["d"],
            input_onnx_file_path=TEST_MODEL_PATH,
            non_verbose=True,
        )
        node_names = [n.name for n in model.graph.node]
        assert "Relu_0" not in node_names
        # Original input 'a' should not appear anywhere
        for node in model.graph.node:
            assert "a" not in list(node.input)

    def test_downstream_nodes_preserved(self):
        """Add_1 and Sigmoid_2 should remain in the extracted model."""
        model = extraction(
            input_op_names=["b"],
            output_op_names=["d"],
            input_onnx_file_path=TEST_MODEL_PATH,
            non_verbose=True,
        )
        node_names = [n.name for n in model.graph.node]
        assert "Add_1" in node_names
        assert "Sigmoid_2" in node_names

    def test_output_is_correct(self):
        """Extracted model output should be 'd'."""
        model = extraction(
            input_op_names=["b"],
            output_op_names=["d"],
            input_onnx_file_path=TEST_MODEL_PATH,
            non_verbose=True,
        )
        output_names = [out.name for out in model.graph.output]
        assert output_names == ["d"]

    def test_passes_onnx_checker(self):
        """Extracted model should be valid ONNX."""
        model = extraction(
            input_op_names=["b"],
            output_op_names=["d"],
            input_onnx_file_path=TEST_MODEL_PATH,
            non_verbose=True,
        )
        onnx.checker.check_model(model)
