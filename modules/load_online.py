import json
import os

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

demos_json = read_json_file("online/demos.json")
tools_json = read_json_file("online/tools.json")

demos_dict = {}
for name, values in demos_json.items():
    demos_dict[name] = values
tools_dict = {}
for name, values in tools_json.items():
    tools_dict[name] = values

demos_names = []
tools_names = []

def load_demos_names():
    demos_names.extend(demos_dict.keys())
    return demos_names

def load_tools_names():
    tools_names.extend(tools_dict.keys())
    return tools_names

def load_demos_url(name):
    return demos_dict[name][0]

def load_tools_url(name):
    return tools_dict[name][0]
