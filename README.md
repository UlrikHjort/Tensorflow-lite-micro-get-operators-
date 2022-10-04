Tool to get the operators used in a trained model tflite file. Intended usage is to find the subset of operators used by the tflite model when optimizing tflite micro codesize.
The tool provide a list of used operators which can be used with MicroMutableOpResolver (add the ops in list)  - or with AllOpsResolver (Remove the ops not in list)

Usage: python get_ops.py <tflite model>

Print out a list of used operators

Requirements: Tensorflow version >= 2.5

Inspiration and code snippets from the tensorflow visualize.py tool.