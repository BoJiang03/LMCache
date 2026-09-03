"""Opt-in probe on vLLM's forward pass, installed independently of any connector.

WHY THIS EXISTS
---------------
The +5.7 ms/step LMCache costs is not in any KVConnector hook: all 20 are timed
(1j/1k), the scheduler's total 8.5 ms/step and the workers' 0.75, and the
scheduler's largest one turned out to be free (1l).  So the work is somewhere
vLLM never calls into, and the next question is not *which hook* but *which
side*:

    the GPU got slower   -- LMCache's copies contend for copy engines, SMs, or
                            the fabric the TP all-reduces use
    the GPU did the same work and the CPU took longer to drive it
                         -- LMCache's threads hold the GIL in the worker
                            process; each worker burns 187% of a core with
                            LMCache attached and only one thread runs bytecode

WHAT IT MEASURES, per model-runner step, per worker

    loop  wall between successive entries of execute_model -- the whole step
    exec  wall inside execute_model
    cpu   thread CPU inside execute_model, same thread

`exec - cpu` is the time the worker's main thread sat blocked rather than
running bytecode.  execute_model launches asynchronously but ends by touching
device results, so that block is the GPU.  Therefore:

    exec-cpu grows with LMCache attached  -> the device is doing more work
    cpu grows, exec-cpu flat              -> the cost is host-side, the GIL

NO CUDA CALLS.  The first version of this file wrapped each step in a
torch.cuda.Event pair to read GPU time directly.  It installed in all 8
workers, survived model load and CUDA graph capture, and then killed
Worker_TP1 on the first inference step with no Python traceback -- a hard
process death, not an exception, which is what recording a timing event into a
capturing stream does.  wall-minus-thread-CPU answers the same question using
nothing but time.perf_counter and time.thread_time, so this version makes no
CUDA API call at all and cannot repeat that.

WHY NOT A CONNECTOR SUBCLASS.  The baseline arm has no connector, so the
instrument cannot live in one or the comparison has no zero.  A sitecustomize
on PYTHONPATH is imported by the interpreter before anything else, in the API
server, the EngineCore and every TP worker alike.

Guarded by STEPPROBE=1: unset, this file defines nothing and returns, so a
stray PYTHONPATH cannot perturb an uninstrumented run.  Every step is wrapped
in try/except -- a probe that breaks the process it measures is worse than no
probe, and this one is imported by every Python process that inherits the path.
"""

import os

if os.environ.get("STEPPROBE") == "1":
    import importlib.util
    import sys
    import time

    _TARGET = "vllm.v1.worker.gpu_model_runner"
    _EVERY = int(os.environ.get("STEPPROBE_EVERY", "200"))

    class _PostImport:
        """Run a callback right after one module finishes executing.

        A plain meta_path finder returns a spec and never learns when the module
        body finished; wrapping the loader's exec_module does, which is the only
        place a class defined by that module can be patched before anything
        instantiates it.
        """

        def __init__(self, target, cb):
            self.target, self.cb = target, cb

        def find_spec(self, fullname, path=None, target=None):
            if fullname != self.target:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module):
                orig(module)
                try:
                    self.cb(module)
                except Exception as e:  # never break the import being probed
                    print(f"STEPPROBE install failed: {e}", file=sys.stderr)

            loader.exec_module = exec_module
            return spec

    # STEPPROBE_CPROFILE="start:stop" runs cProfile around execute_model for
    # that window of steps in ONE worker and writes pstats to
    # $STEPPROBE_CPROFILE_OUT.  Reading LMCache's source has now produced six
    # mechanisms and had all six refuted by experiment; the remaining 67 ms/step
    # inside IP's wait_for_save is not attributable to any statement visible in
    # that function, so the next step is to measure which function it is rather
    # than to guess again.  Off unless the variable is set.
    _PROF_WINDOW = os.environ.get("STEPPROBE_CPROFILE", "")
    _PROF_OUT = os.environ.get("STEPPROBE_CPROFILE_OUT", "/tmp/stepprobe")

    def _install(module):
        from vllm.logger import init_logger

        logger = init_logger("vllm.stepprobe")
        runner = module.GPUModelRunner
        orig_execute = runner.execute_model
        st = {"steps": 0, "wall": 0.0, "cpu": 0.0, "t0": time.perf_counter()}

        prof = None
        p_start = p_stop = -1
        if _PROF_WINDOW:
            try:
                import cProfile
                a, b = _PROF_WINDOW.split(":")
                p_start, p_stop = int(a), int(b)
                prof = cProfile.Profile()
                logger.info("STEPPROBE cProfile armed for steps %d..%d pid=%d -> %s",
                            p_start, p_stop, os.getpid(), _PROF_OUT)
            except Exception as e:
                logger.warning("STEPPROBE cProfile could not arm: %s", e)
                prof = None
        # Every worker dumps its own file, named by pid, so four ranks cannot
        # clobber one another's stats.

        def probed(self, *a, **kw):
            t0, c0 = time.perf_counter(), time.thread_time()
            if prof is not None and st["steps"] == p_start:
                try:
                    prof.enable()
                    logger.info("STEPPROBE cProfile ON at step %d pid=%d",
                                st["steps"], os.getpid())
                except Exception:
                    pass
            try:
                return orig_execute(self, *a, **kw)
            finally:
                try:
                    st["steps"] += 1
                    st["wall"] += time.perf_counter() - t0
                    st["cpu"] += time.thread_time() - c0
                    n = st["steps"]
                    if prof is not None and n == p_stop:
                        try:
                            prof.disable()
                            prof.dump_stats(f"{_PROF_OUT}.{os.getpid()}.pstats")
                            logger.info("STEPPROBE cProfile OFF at step %d, "
                                        "stats -> %s.%d.pstats", n, _PROF_OUT,
                                        os.getpid())
                        except Exception as e:
                            logger.warning("STEPPROBE cProfile dump failed: %s", e)
                    if n % _EVERY == 0:
                        dw = time.perf_counter() - st["t0"]
                        logger.info(
                            "STEPPROBE pid=%d steps=%d loop_ms/step=%.3f "
                            "exec_wall_ms/step=%.3f exec_cpu_ms/step=%.3f",
                            os.getpid(), n, 1000 * dw / n,
                            1000 * st["wall"] / n, 1000 * st["cpu"] / n,
                        )
                except Exception:
                    pass

        probed.__name__ = "execute_model"
        probed.__qualname__ = "GPUModelRunner.execute_model"
        probed.__doc__ = orig_execute.__doc__
        runner.execute_model = probed
        logger.info("STEPPROBE installed on GPUModelRunner.execute_model pid=%d",
                    os.getpid())

    try:
        sys.meta_path.insert(0, _PostImport(_TARGET, _install))
    except Exception as e:
        print(f"STEPPROBE could not arm: {e}", file=sys.stderr)
