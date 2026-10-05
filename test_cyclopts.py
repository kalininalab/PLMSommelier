import cyclopts

app = cyclopts.App()
@app.command
def foo():
    pass
cmd, bound, _ = app.parse_args([])
print("cmd:", cmd)
print("bound:", bound)
res = cmd(*bound.args, **bound.kwargs)
print("res:", res)
