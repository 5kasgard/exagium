import ast, sys, subprocess
files = subprocess.run(["git","ls-files","*.py"],capture_output=True,text=True,cwd=sys.argv[1]).stdout.split()
for f in files:
    if f.startswith(("loop/results","verifications","runs")): continue
    src = open(f"{sys.argv[1]}/{f}").read()
    try: t = ast.parse(src)
    except SyntaxError: continue
    lines = src.split("\n")
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            doc = (ast.get_docstring(n) or "").split("\n")[0][:100]
            body = " | ".join(l.strip() for l in lines[n.lineno:n.lineno+4] if l.strip() and not l.strip().startswith(('"""',"#")))[:220]
            print(f"{f}:{n.lineno} def {n.name}({', '.join(a.arg for a in n.args.args)})  # {doc}  :: {body}")
