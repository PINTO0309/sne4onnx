# sne4onnx
A very simple tool for situations where optimization with onnx-simplifier would exceed the Protocol Buffers upper file size limit of 2GB, or simply to separate onnx files to any size you want. **S**imple **N**etwork **E**xtraction for **ONNX**.

# Key concept
- [ ] If INPUT OP name and OUTPUT OP name are specified, the onnx graph within the range of the specified OP name is extracted and .onnx is generated.
