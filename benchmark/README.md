benchmodulellm can be used to test llm unit inference performance

Only the llm unit definition files (model json) are required.

If no model specified, it would benchmark default list. More model networks may be added later.

Usage
```shell
python benchmodulellm.py --host 192.168.20.100 --port 10001 --test-items default.yaml
```