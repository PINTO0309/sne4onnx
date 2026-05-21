"""Generate a small ONNX model for testing mid-graph tensor extraction.

Model structure:
  input 'a' [1,4] -> Relu -> output 'b' [1,4] -> Add(b, const) -> output 'c' [1,4] -> Sigmoid -> output 'd' [1,4]

Test scenario: extract sub-graph with input='b', output='d'
Expected: extracted model starts at 'b', does not contain the Relu node that produces 'b'.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

def generate_test_model(output_path: str = "tests/test_mid_graph_extraction.onnx"):
    # Nodes: a -> Relu -> b -> Add(b, ones) -> c -> Sigmoid -> d
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 4])
    d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1, 4])

    ones = helper.make_tensor("ones", TensorProto.FLOAT, [1, 4], np.ones([1, 4]).flatten().tolist())

    relu_node = helper.make_node("Relu", inputs=["a"], outputs=["b"], name="Relu_0")
    add_node = helper.make_node("Add", inputs=["b", "ones"], outputs=["c"], name="Add_1")
    sigmoid_node = helper.make_node("Sigmoid", inputs=["c"], outputs=["d"], name="Sigmoid_2")

    graph = helper.make_graph(
        [relu_node, add_node, sigmoid_node],
        "test_mid_graph",
        [a],
        [d],
        initializer=[ones],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved test model to {output_path}")

if __name__ == "__main__":
    generate_test_model()
