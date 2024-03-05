import multiprocessing as mp
import functools
import traceback
import sys
from typing import Tuple, Any
import datetime

def subprocess_worker_fn(worker_id, conn: 'mp.connection.Connection', fn, verbose=False):
  """Args:
     conn: one end of mp.Pipe()
  """
  # {"command": ,"args": args, "kwargs": kwargs}
  if verbose:
    print(f"Subprocess worker {worker_id} started")
  while True:
    cmd = conn.recv()
    if cmd["command"] == "exit":
      if verbose:
        print(f"Subprocess worker {worker_id} received exiting command, exit")
      conn.close()
      break
    elif cmd["command"] == "run":
      # print("my cmd")
      errorReason = None

      try:
        res = fn(*cmd["args"], **cmd["kwargs"])
      except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        serialized_exception = traceback.format_exception(exc_type, exc_value, exc_traceback)
        errorReason = "".join(serialized_exception)
        res = None
      finally:
        conn.send({
          "result": res,
          "errorReason": errorReason
        })
    else:
      raise Exception(f"Subprocess worker {worker_id} received unknown command, abort")


"""
# https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
from multiprocessing import Pool


_func = None

def worker_init(func):
  global _func
  _func = func
  

def worker(x):
  return _func(x)


with Pool(None, initializer=worker_init, initargs=(lambda x: x ** 2,)) as p:
  print(p.map(worker, range(100)))

# https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
"""

class SubprocessManagerMock:
  def __init__(self, destFn, worker_idx_begin=0, verbose=False):
    self.destFn = destFn
  
  def spawnExec(self, *args, **kwargs):
    return self.destFn(*args, **kwargs), True, None
  
  def endExec(self):
    pass

ERROR_REASON_CRASH_OR_TOO_LONG_TO_RESPOND = "Subprocess crashed or took too long to respond"

# Used to circumvent engine TDR
class SubprocessManager:
  """Pool with single worker and recovers on process abnormally exits"""
  def __init__(self, destFn, worker_idx_begin=0, verbose=False, timeout=None):
    """Timeout is in seconds
    NOTE: the child of the child process will not be killed, since they don't belong to a process group.
    TODO for this small library is to include such situation. 
    """
    self.mp_ctx = mp.get_context('spawn')
    self.next_worker_idx = worker_idx_begin
    self.verbose = verbose
    self.timeout = timeout
    
    self.destFn = destFn
    self.p = None
    self.pipe_conn = None
    self.respawn_worker()

  def respawn_worker(self):
    # print(self.p.is_alive()) if self.p is not None else None
    assert(self.p is None or (not self.p.is_alive()))
    server_side, client_side = self.mp_ctx.Pipe()
    self.p = self.mp_ctx.Process(target=subprocess_worker_fn, args=[
      self.next_worker_idx,
      client_side,
      self.destFn,
      self.verbose
    ])
    self.pipe_conn = server_side
    self.next_worker_idx += 1
    
    self.p.start()

  def spawnExec(self, *args, **kwargs) -> Tuple[Any, bool, str]:
    """Returns (result, run_successful, error_reason)"""
    cmd = {
      'command': 'run',
      'args': args,
      'kwargs': kwargs
    }
    res = None

    run_failed = False
    error_reason = None
    # print(cmd)
    self.pipe_conn.send(cmd)
    try:

      # If timeout is None then an infinite timeout is used
      dataAvailable = self.pipe_conn.poll(timeout=self.timeout)
      if not dataAvailable:
        # TIMEOUT! Kill subprocess and continue
        run_failed = True
      else:
        cmd_reply = self.pipe_conn.recv()
        res = cmd_reply["result"]
        error_reason = cmd_reply["errorReason"]

    except EOFError:
      # failed to run this command
      run_failed = True

    if run_failed:
      # TODO: give timeout
      self.p.kill()
      self.p.join()
      self.respawn_worker()
      error_reason = ERROR_REASON_CRASH_OR_TOO_LONG_TO_RESPOND
    
    return res, not (run_failed or error_reason is not None), error_reason

  def endExec(self):
    if self.p is None or (not self.p.is_alive()):
      return
    
    self.pipe_conn.send({
      'command': 'exit'
    })

def testFunc(a, b):
  return a + b

def testAbnormalFunc(a, b):
  if a == 1 and b == 1:
    raise Exception("test")
  else:
    return a + b

def testTimeoutFunc(a):
  while True:
    a += 1

if __name__ == '__main__':
  spm1 = SubprocessManager(testFunc)
  # print(1)
  res = spm1.spawnExec(a=1, b=1)
  
  print(f"sp1: testFunc(a=1, b=1) returns {res}")
  spm1.endExec()

  spm2 = SubprocessManager(testAbnormalFunc, worker_idx_begin=1)
  res = spm2.spawnExec(a=1, b=1)
  print(f"sp2: testAbnormalFunc(a=1, b=1) returns {res}")

  res = spm2.spawnExec(a=1, b=2)
  print(f"sp2: testAbnormalFunc(a=1, b=2) returns {res}")
  res = spm2.spawnExec(a=1, b=3)
  print(f"sp2: testAbnormalFunc(a=1, b=3) returns {res}")
  spm2.endExec()

  # test partial binding of global func
  spm3 = SubprocessManager(functools.partial(testFunc, a=1), worker_idx_begin=1)
  res = spm3.spawnExec(b=2)
  print(f"sp3: functools.partial(testFunc, a=1)(b=2) returns {res}")
  spm3.endExec()

  # test timeout
  print(f"Now: {datetime.datetime.now()}")
  spm4 = SubprocessManager(testTimeoutFunc, worker_idx_begin=1, verbose=True, timeout=2)
  res = spm4.spawnExec(a = 1)
  print(f"sp4: testTimeoutFunc(a=1) returns {res}")
  spm4.endExec()
  print(f"Now: {datetime.datetime.now()}")

  # lambda and local functions are known to fail.
  # TODO: fix this
