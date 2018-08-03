

```python
import numba
print(numba.__version__)
```

    0.35.0+10.g143f70e
    


```python
def bubblesort(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
```


```python
import numpy as np

original = np.arange(0.0, 10.0, 0.01, dtype='f4')
shuffled = original.copy()
np.random.shuffle(shuffled)
```


```python
sorted = shuffled.copy()
bubblesort(sorted)
print(np.array_equal(sorted, original))
```

    True
    


```python
sorted[:] = shuffled[:]
%timeit sorted[:] = shuffled[:]; bubblesort(sorted)
```

    141 ms ± 3.99 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    


```python
print(numba.jit.__doc__)
```

    
        This decorator is used to compile a Python function into native code.
    
        Args
        -----
        signature:
            The (optional) signature or list of signatures to be compiled.
            If not passed, required signatures will be compiled when the
            decorated function is called, depending on the argument values.
            As a convenience, you can directly pass the function to be compiled
            instead.
    
        locals: dict
            Mapping of local variable names to Numba types. Used to override the
            types deduced by Numba's type inference engine.
    
        target: str
            Specifies the target platform to compile for. Valid targets are cpu,
            gpu, npyufunc, and cuda. Defaults to cpu.
    
        options:
            For a cpu target, valid options are:
                nopython: bool
                    Set to True to disable the use of PyObjects and Python API
                    calls. The default behavior is to allow the use of PyObjects
                    and Python API. Default value is False.
    
                forceobj: bool
                    Set to True to force the use of PyObjects for every value.
                    Default value is False.
    
                looplift: bool
                    Set to True to enable jitting loops in nopython mode while
                    leaving surrounding code in object mode. This allows functions
                    to allocate NumPy arrays and use Python objects, while the
                    tight loops in the function can still be compiled in nopython
                    mode. Any arrays that the tight loop uses should be created
                    before the loop is entered. Default value is True.
    
                error_model: str
                    The error-model affects divide-by-zero behavior.
                    Valid values are 'python' and 'numpy'. The 'python' model
                    raises exception.  The 'numpy' model sets the result to
                    *+/-inf* or *nan*.
    
        Returns
        --------
        A callable usable as a compiled function.  Actual compiling will be
        done lazily if no explicit signatures are passed.
    
        Examples
        --------
        The function can be used in the following ways:
    
        1) jit(signatures, target='cpu', **targetoptions) -> jit(function)
    
            Equivalent to:
    
                d = dispatcher(function, targetoptions)
                for signature in signatures:
                    d.compile(signature)
    
            Create a dispatcher object for a python function.  Then, compile
            the function with the given signature(s).
    
            Example:
    
                @jit("int32(int32, int32)")
                def foo(x, y):
                    return x + y
    
                @jit(["int32(int32, int32)", "float32(float32, float32)"])
                def bar(x, y):
                    return x + y
    
        2) jit(function, target='cpu', **targetoptions) -> dispatcher
    
            Create a dispatcher function object that specializes at call site.
    
            Examples:
    
                @jit
                def foo(x, y):
                    return x + y
    
                @jit(target='cpu', nopython=True)
                def bar(x, y):
                    return x + y
    
        
    


```python
bubblesort_jit = numba.jit("void(f4[:])")(bubblesort)
```


```python
sorted[:] = shuffled[:] # reset to shuffled before sorting
bubblesort_jit(sorted)
print(np.array_equal(sorted, original))
```

    True
    


```python
%timeit sorted[:] = shuffled[:]; bubblesort_jit(sorted)
```

    571 µs ± 2.39 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    


```python
%timeit sorted[:] = shuffled[:]; bubblesort(sorted)
```

    137 ms ± 1.43 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    


```python
@numba.jit("void(f4[:])")
def bubblesort_jit(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
```


```python
bubblesort_autojit = numba.jit(bubblesort)
```


```python
%timeit sorted[:] = shuffled[:]; bubblesort_autojit(sorted)
```

    988 µs ± 2.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    


```python
import sys
@numba.jit("void(i1[:])")
def test(value):
    for i in range(len(value)):
        value[i] = i % 100

from decimal import Decimal
@numba.jit("void(i1[:])")
def test2(value):
    for i in range(len(value)):
        value[i] = i % Decimal(100)

res = np.zeros((10000,), dtype="i1")
```

**Note :** xrange was in python 2. In python 3 it was renamed to range. [ref](https://stackoverflow.com/questions/17192158/nameerror-global-name-xrange-is-not-defined-in-python-3)


```python
%timeit test(res)
```

    12.3 µs ± 72.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    


```python
%timeit test2(res)
```

    3.86 ms ± 18.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    


```python
@numba.jit("void(i1[:])", nopython=True)
def test(value):
    for i in range(len(value)):
        value[i] = i % 100
```


```python
@numba.jit("void(i1[:])", nopython=True)
def test2(value):
    for i in range(len(value)):
        value[i] = i % Decimal(100)
```


    ---------------------------------------------------------------------------

    TypingError                               Traceback (most recent call last)

    <ipython-input-61-5fae05749356> in <module>()
    ----> 1 @numba.jit("void(i1[:])", nopython=True)
          2 def test2(value):
          3     for i in range(len(value)):
          4         value[i] = i % Decimal(100)
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\decorators.py in wrapper(func)
        197             with typeinfer.register_dispatcher(disp):
        198                 for sig in sigs:
    --> 199                     disp.compile(sig)
        200                 disp.disable_compile()
        201         return disp
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\dispatcher.py in compile(self, sig)
        577 
        578                 self._cache_misses[sig] += 1
    --> 579                 cres = self._compiler.compile(args, return_type)
        580                 self.add_overload(cres)
        581                 self._cache.save_overload(sig, cres)
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\dispatcher.py in compile(self, args, return_type)
         78                                       impl,
         79                                       args=args, return_type=return_type,
    ---> 80                                       flags=flags, locals=self.locals)
         81         # Check typing error if object mode is used
         82         if cres.typing_error is not None and not flags.enable_pyobject:
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in compile_extra(typingctx, targetctx, func, args, return_type, flags, locals, library)
        761     pipeline = Pipeline(typingctx, targetctx, library,
        762                         args, return_type, flags, locals)
    --> 763     return pipeline.compile_extra(func)
        764 
        765 
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in compile_extra(self, func)
        358         self.lifted = ()
        359         self.lifted_from = None
    --> 360         return self._compile_bytecode()
        361 
        362     def compile_ir(self, func_ir, lifted=(), lifted_from=None):
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in _compile_bytecode(self)
        720         """
        721         assert self.func_ir is None
    --> 722         return self._compile_core()
        723 
        724     def _compile_ir(self):
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in _compile_core(self)
        707 
        708         pm.finalize()
    --> 709         res = pm.run(self.status)
        710         if res is not None:
        711             # Early pipeline completion
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in run(self, status)
        244                     # No more fallback pipelines?
        245                     if is_final_pipeline:
    --> 246                         raise patched_exception
        247                     # Go to next fallback pipeline
        248                     else:
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in run(self, status)
        236                 try:
        237                     event(stage_name)
    --> 238                     stage()
        239                 except _EarlyPipelineCompletion as e:
        240                     return e.result
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in stage_nopython_frontend(self)
        450                 self.args,
        451                 self.return_type,
    --> 452                 self.locals)
        453 
        454         with self.fallback_context('Function "%s" has invalid return type'
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\compiler.py in type_inference_stage(typingctx, interp, args, return_type, locals)
        862             infer.seed_type(k, v)
        863 
    --> 864         infer.build_constraint()
        865         infer.propagate()
        866         typemap, restype, calltypes = infer.unify()
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\typeinfer.py in build_constraint(self)
        798         for blk in utils.itervalues(self.blocks):
        799             for inst in blk.body:
    --> 800                 self.constrain_statement(inst)
        801 
        802     def return_types_from_partial(self):
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\typeinfer.py in constrain_statement(self, inst)
        957     def constrain_statement(self, inst):
        958         if isinstance(inst, ir.Assign):
    --> 959             self.typeof_assign(inst)
        960         elif isinstance(inst, ir.SetItem):
        961             self.typeof_setitem(inst)
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\typeinfer.py in typeof_assign(self, inst)
       1019                                               src=value.name, loc=inst.loc))
       1020         elif isinstance(value, (ir.Global, ir.FreeVar)):
    -> 1021             self.typeof_global(inst, inst.target, value)
       1022         elif isinstance(value, ir.Arg):
       1023             self.typeof_arg(inst, inst.target, value)
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\typeinfer.py in typeof_global(self, inst, target, gvar)
       1115     def typeof_global(self, inst, target, gvar):
       1116         try:
    -> 1117             typ = self.resolve_value_type(inst, gvar.value)
       1118         except TypingError as e:
       1119             if (gvar.name == self.func_id.func_name
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\numba\typeinfer.py in resolve_value_type(self, inst, val)
       1038         except ValueError as e:
       1039             msg = str(e)
    -> 1040         raise TypingError(msg, loc=inst.loc)
       1041 
       1042     def typeof_arg(self, inst, target, arg):
    

    TypingError: Failed at nopython (nopython frontend)
    Untyped global name 'Decimal': cannot determine Numba type of <class 'type'>
    File "<ipython-input-61-5fae05749356>", line 4

