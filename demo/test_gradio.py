#!/usr/bin/env python3
print("Step 1: Starting script")

import sys
print("Step 2: Imported sys")

import gradio as gr
print(f"Step 3: Imported Gradio version {gr.__version__}")

def hello(name):
    return f"Hello {name}!"

print("Step 4: Creating interface")
demo = gr.Interface(fn=hello, inputs="text", outputs="text")

print("Step 5: Interface created")
print("Step 6: Launching...")

demo.launch(server_port=7863)

print("Step 7: Launch called")