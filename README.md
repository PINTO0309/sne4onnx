# sne4onnx
A very simple tool for situations where optimization with onnx-simplifier would exceed the Protocol Buffers upper file size limit of 2GB, or simply to separate onnx files to any size you want. **S**imple **N**etwork **E**xtraction for **ONNX**.

[![Downloads](https://static.pepy.tech/personalized-badge/sne4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sne4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sne4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sne4onnx?color=2BAF2B)](https://pypi.org/project/sne4onnx/) [![CodeQL](https://github.com/PINTO0309/sne4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sne4onnx/actions?query=workflow%3ACodeQL)

# Key concept
- [x] If INPUT OP name and OUTPUT OP name are specified, the onnx graph within the range of the specified OP name is extracted and .onnx is generated.
- [x] Change backend to `onnx.utils.Extractor.extract_model` so that onnx.ModelProto can be specified as input.

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& pip install -U sne4onnx
```
### 1-2. Docker
```bash
### docker pull
$ docker pull pinto0309/sne4onnx:latest

### docker build
$ docker build -t pinto0309/sne4onnx:latest .

### docker run
$ docker run --rm -it -v `pwd`:/workdir pinto0309/sne4onnx:latest
$ cd /workdir
```

## 2. CLI Usage
```bash
$ sne4onnx -h

usage:
    sne4onnx [-h]
    --input_onnx_file_path INPUT_ONNX_FILE_PATH
    --input_op_names INPUT_OP_NAMES
    --output_op_names OUTPUT_OP_NAMES
    [--output_onnx_file_path OUTPUT_ONNX_FILE_PATH]

optional arguments:
  -h, --help
        show this help message and exit

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
        Input onnx file path.

  --input_op_names INPUT_OP_NAMES
        List of OP names to specify for the input layer of the model.
        Specify the name of the OP, separated by commas.
        e.g. --input_op_names aaa,bbb,ccc

  --output_op_names OUTPUT_OP_NAMES
        List of OP names to specify for the output layer of the model.
        Specify the name of the OP, separated by commas.
        e.g. --output_op_names ddd,eee,fff

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path. If not specified, extracted.onnx is output.
```

## 3. In-script Usage
```bash
$ python
>>> from sne4onnx import extraction
>>> help(extraction)

Help on function extraction in module sne4onnx.onnx_network_extraction:

extraction(
    input_onnx_file_path: str,
    input_op_names: List[str],
    output_op_names: List[str],
    output_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: str
        Input onnx file path.

    input_op_names: List[str]
        List of OP names to specify for the input layer of the model.
        Specify the name of the OP, separated by commas.
        e.g. ['aaa','bbb','ccc']

    output_op_names: List[str]
        List of OP names to specify for the output layer of the model.
        Specify the name of the OP, separated by commas.
        e.g. ['ddd','eee','fff']

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If not specified, .onnx is not output.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    Returns
    -------
    extracted_graph: onnx.ModelProto
        Extracted onnx ModelProto
```

## 4. CLI Execution
```bash
$ sne4onnx \
--input_onnx_file_path input.onnx \
--input_op_names aaa,bbb,ccc \
--output_op_names ddd,eee,fff \
--output_onnx_file_path output.onnx
```

## 5. In-script Execution
### 5-1. Use ONNX files
```python
from sne4onnx import extraction

extracted_graph = extraction(
  input_onnx_file_path='input.onnx',
  input_op_names=['aaa', 'bbb', 'ccc'],
  output_op_names=['ddd', 'eee', 'fff'],
  output_onnx_file_path='output.onnx',
)
```
### 5-2. Use onnx.ModelProto
```python
from sne4onnx import extraction

extracted_graph = extraction(
  input_op_names=['aaa', 'bbb', 'ccc'],
  output_op_names=['ddd', 'eee', 'fff'],
  output_onnx_file_path='output.onnx',
  onnx_graph=graph,
)
```

## 6. Samples
### 6-1. Pre-extraction
![image](https://user-images.githubusercontent.com/33194443/162101010-13662cb6-a93b-4ebb-ad46-96da055a56a4.png)
![image](https://user-images.githubusercontent.com/33194443/162100392-71d58154-ea75-4a39-88a5-930a6e7a5d6a.png)
![image](https://user-images.githubusercontent.com/33194443/162100741-89e5cf0e-de21-469c-a060-1a05a3a2ce1b.png)

### 6-2.  Extraction
```bash
$ sne4onnx \
--input_onnx_file_path hitnet_sf_finalpass_720x1280.onnx \
--input_op_names 0,1 \
--output_op_names 497,785 \
--output_onnx_file_path hitnet_sf_finalpass_720x960_head.onnx
```

### 6-3. Extracted
![image](https://user-images.githubusercontent.com/33194443/162101435-a9e1209b-8b87-4c85-b66e-517e26aab9ba.png)
![image](https://user-images.githubusercontent.com/33194443/162101596-ba0cd103-3daa-4a2b-98d4-cf4d72074f64.png)
![image](https://user-images.githubusercontent.com/33194443/162101783-45e0fde7-2d9a-4625-a0f8-95efa7f79473.png)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/snd4onnx
5. https://github.com/PINTO0309/scs4onnx
6. https://github.com/PINTO0309/snc4onnx
