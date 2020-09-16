## {{{ docstring

"""
sm.py - state machine library

tested with:
    - python 2.6.6
    - pypy   2.7.13
    - python 2.7.16
    - python 2.7.18
    - pypy   3.5.3
    - python 3.6.12
    - python 3.7.3
    - python 3.8.5
"""

## }}}
## {{{ prologue
from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace
## XXX pylint: disable=missing-docstring

import collections
import sys
import weakref

__all__ = [
    "AllSymbols",
    "EofSymbol",
    "SM",
    "error",
    "programming_error",
    "sm_error",
]

OLD_PY = sys.version_info[0] < 3

## }}}
## {{{ support classes and functions

class _Absent(object):
    ## pylint: disable=too-few-public-methods
    pass

class AllSymbols(object):
    ## pylint: disable=too-few-public-methods
    pass

class EofSymbol(object):
    ## pylint: disable=too-few-public-methods
    pass

class sm_error(SyntaxError):
    ## pylint: disable=too-few-public-methods
    pass

class error(sm_error, SyntaxError):
    ## pylint: disable=too-few-public-methods
    pass

class programming_error(sm_error, RuntimeError):
    ## pylint: disable=too-few-public-methods
    pass

def base_with_mcs(mcs):
    meta_name  = mcs.__name__
    class_name = "class_%s" % meta_name
    if OLD_PY:
        template = "class %s(object):\n __metaclass__ = %s"
    else:
        template = "class %s(metaclass=%s):\n pass"
    src  = template % (class_name, meta_name)
    code = compile(src, class_name, "exec")
    glb  = { meta_name: mcs }
    eval(code, glb) ## pylint: disable=eval-used
    return glb[class_name]

_DeferredStateInfo = collections.namedtuple(
    "DeferredStateInfo",
    "start_state symbol function end_state"
)

_DeferredStateFunctionAttr = "___deferred_sm_state_info___"

## }}}
## {{{ transition() method decorator

def transition(
        start_state, symbol, end_state=_Absent
    ):
    def transition_wrapper(function):
        transitions = getattr(function, _DeferredStateFunctionAttr, None)
        if transitions is None:
            transitions = [ ]
            setattr(function, _DeferredStateFunctionAttr, transitions)
        transitions.insert(
            0,
            _DeferredStateInfo(
                start_state, symbol, function, end_state
            )
        )
        return function
    return transition_wrapper

## }}}
## {{{ sm metaclass

class _SMMetaclass(type):
    _class_transition_map = weakref.WeakKeyDictionary()

    def __new__(mcs, name, bases, attrs):
        cls  = type.__new__(mcs, name, bases, attrs)
        xits = mcs._class_transition_map.setdefault(cls, { })
        for a, v in attrs.items():
            if not callable(v):
                continue
            deferred = getattr(v, _DeferredStateFunctionAttr, None)
            if deferred is None:
                continue
            delattr(v, _DeferredStateFunctionAttr)
            xits[a] = deferred
        return cls

    def __call__(cls, *args, **kw):
        obj = type.__call__(cls, *args, **kw)
        if not callable(getattr(obj, "add_transition", None)):
            return obj
        xits = type(cls)._class_transition_map[cls]
        for attr, state_info in xits.items():
            func = getattr(obj, attr, None)
            func = getattr(func, "__func__", func)
            if not callable(func):
                continue
            for info in state_info:
                if info.function is not func:
                    continue
                obj.add_transition(
                    start_state=info.start_state,
                    symbol=info.symbol,
                    function=info.function,
                    end_state=info.end_state,
                )
        return obj

## }}}
## {{{ sm base class

class _SMBase(base_with_mcs(_SMMetaclass)):
    _RUN_STATE_INIT = 0
    _RUN_STATE_RUN  = 1
    _RUN_STATE_DONE = 2
    _RUN_STATE_DEAD = 3

    def __init__(self):
        self._runstate = self._RUN_STATE_INIT

    ########################################
    ## public methods

    def add_transition(
            self,
            start_state, symbol, function=_Absent, end_state=_Absent
        ):
        if self._runstate != self._RUN_STATE_INIT:
            self._error("sm already initialized")
        if isinstance(symbol, list):
            if not symbol:
                self._error("no symbols provided")
            for sym in symbol:
                self.add_transition(start_state, sym, function, end_state)
            return
        if not self._check_state(start_state):
            self._error("bad start_state")
        if not self._check_symbol(symbol):
            self._error("bad symbol")
        if function is _Absent:
            function = lambda *_: None
        elif not callable(function):
            self._error("function not callable")
        if not (end_state is _Absent or self._check_state(end_state)):
            self._error("bad end_state")
        self._add_transition(
            start_state, symbol, function, end_state
        )

    def status(self):
        ## pylint: disable=no-self-use
        return None

    def running(self):
        rs = self._runstate
        if rs == self._RUN_STATE_DONE:
            self._error("sm is done")
        elif rs == self._RUN_STATE_DEAD:
            self._error("sm is dead")
        return rs == self._RUN_STATE_RUN

    def start(self):
        rs = self._runstate
        if rs == self._RUN_STATE_RUN:
            self._error("sm already initialized")
        elif rs == self._RUN_STATE_DONE:
            self._error("sm is done")
        elif rs == self._RUN_STATE_DEAD:
            self._error("sm is dead")
        self._runstate = self._RUN_STATE_RUN

    def started(self):
        return self._runstate == self._RUN_STATE_RUN

    def stopped(self):
        return self._runstate == self._RUN_STATE_DONE

    def transition( ## pylint: disable=too-many-arguments
            self, start_state, symbol, function=_Absent, end_state=_Absent
        ):
        if function is _Absent:
            def transition_wrapper(function):
                self.add_transition(
                    start_state, symbol, function, end_state
                )
                return function
            return transition_wrapper
        self.add_transition(
            start_state, symbol, function, end_state
        )
        return function

    ########################################
    ## subclasses must override these

    def _add_transition( ## pylint: disable=too-many-arguments
            self, start_state, symbol, function=_Absent, end_state=_Absent
        ):
        raise NotImplementedError()

    def _check_state(self, state):
        ## pylint: disable=unused-argument,no-self-use
        raise NotImplementedError()

    def _check_symbol(self, symbol):
        ## pylint: disable=unused-argument,no-self-use
        raise NotImplementedError()

    ########################################
    ## internal methods

    def _check_run(self):
        rs = self._runstate
        if rs == self._RUN_STATE_INIT:
            self._error("sm has not started")
        elif rs == self._RUN_STATE_DONE:
            self._error("sm is done")
        elif rs == self._RUN_STATE_DEAD:
            self._error("sm is dead")

    def _error(self, msg):
        self._runstate = self._RUN_STATE_DEAD
        raise programming_error(msg)

    def _stop(self):
        rs = self._runstate
        if rs == self._RUN_STATE_DONE:
            return
        if rs != self._RUN_STATE_RUN:
            self._error("cannot stop sm now")
        self._runstate = self._RUN_STATE_DONE

## }}}
## {{{ sm class

class SM(_SMBase):
    ## pylint: disable=too-many-instance-attributes

    MAX_BUFFER_COUNT   = 32
    MAX_BUFFER_SYMBOLS = 32
    MAX_PUSH_COUNT     = 32

    STATES      = ( )
    START_STATE = _Absent
    STOP_STATES = ( )

    _NOK_STATES = { None: True, _Absent: True }

    def __init__(
            self, states=( ), alphabet=( ),
            start_state=_Absent, stop_states=( ),
        ):
        super(SM, self).__init__()
        states, start_state, stop_states = \
            self._process_states(
                states or self.STATES,
                self.START_STATE if start_state is _Absent else start_state,
                stop_states or self.STOP_STATES
            )
        self._check_args(states, alphabet)

        self._state0  = start_state
        self._stack   = [self._state0]
        self._stops   = dict((s, True) for s in stop_states)
        self._xtable  = { }
        self._feeding = False
        self._buffer  = collections.deque()
        self._bufcnt  = 0
        self._last    = None
        self._symbol  = None
        self._next    = None

    def current(self):
        return self._stack[-1]

    def default(self):  ## pylint: disable=no-self-use
        raise error("no transition defined for current symbol")

    def depth(self):
        return len(self._stack)

    def feed(self, symbol=_Absent):
        self._check_run()
        if isinstance(symbol, list):
            for s in symbol:
                self.feed(s)
            return
        flag = symbol is not _Absent
        if flag:
            if not self._check_symbol(symbol):
                self._error("bad symbol in feed")
        f, self._feeding = self._feeding, True
        if f:
            self._error("already in feed()")
        q = self._buffer
        try:
            if flag:
                self.putback(symbol)
            try:
                while q:
                    self._feed(q.popleft())
                self._bufcnt = 0
            except (KeyboardInterrupt, SystemExit, error):
                raise
            except BaseException as exc:    ## pylint: disable=broad-except
                self._error("feed exception: " + str(exc))
        finally:
            self._feeding = False

    def feed_seq(self, seq):
        self.feed([sym for sym in seq])

    def last(self):
        self._check_run()
        return self._last

    def next(self, symbol):
        self._check_run()
        if not self._check_symbol(symbol):
            self._error("bad symbol in next()")
        self._next = symbol

    def pop(self):
        self._check_run()
        stk = self._stack
        if len(stk) < 2:
            self._error("state stack underflow")
        return stk.pop()

    def push(self, state, return_state=_Absent):
        self._check_run()
        stk = self._stack
        if len(stk) == self.MAX_PUSH_COUNT:
            self._error("state stack overflow")
        if not self._check_state(state):
            self._error("bad push state")
        last = stk[-1]
        if return_state is not _Absent:
            if not self._check_state(return_state):
                self._error("bad return state")
            stk[-1] = return_state
        self._last = last
        stk.append(state)
        return last

    def putback(self, symbol=_Absent):
        self._check_run()
        if symbol is _Absent:
            symbol = self.symbol()
        elif not self._check_symbol(symbol):
            self._error("bad putback symbol")
        q = self._buffer
        if len(q) == self.MAX_BUFFER_SYMBOLS:
            self._error("too many putbacks")
        if self._bufcnt == self.MAX_BUFFER_COUNT:
            self._error("possible putback loop")
        self._bufcnt += 1
        q.append(symbol)

    def putbacks(self):
        return tuple(self._buffer)

    def reset(self):
        self._check_run()
        self._stack   = [self._state0]
        self._buffer  = collections.deque()
        self._bufcnt  = 0
        self._last    = None

    def set_state(self, state):
        if not self._check_state(state):
            self._error("bad state")
        if state in self._stops:
            self.stop()
        stk = self._stack
        old, stk[-1] = stk[-1], state
        return old

    def stop(self):
        self._stop()

    def symbol(self):
        self._check_run()
        if not self._feeding:
            self._error("symbol() called outside of feed")
        return self._symbol

    ########################################

    def _add_transition( ## pylint: disable=too-many-arguments
            self,
            start_state, symbol, function=_Absent, end_state=_Absent
        ):
        tbl = self._xtable.setdefault(start_state, { })
        if symbol in tbl:
            self._error("dup sym handler")
        tbl[symbol] = (function, end_state)

    def _check_args(self, states, alphabet):
        ok  = self._ok_states = { }
        nok = self._NOK_STATES
        for s in states:
            if s in nok:
                self._error("illegal state")
            try:
                ok[s] = True
            except TypeError:
                self._error("unhashable state")
        try:
            alphabet = [s for s in alphabet]
        except ValueError:
            self._error("alphabet must be a sequence")
        ok = self._ok_alpha = { }
        for s in alphabet:
            try:
                ok[s] = True
            except TypeError:
                self._error("unhashable symbol")

    def _check_state(self, state):
        ## pylint: disable=method-hidden
        if state in self._NOK_STATES:
            return False
        ok = self._ok_states
        if ok:
            try:
                return state in ok
            except TypeError:
                return False
            return True
        return True

    def _check_symbol(self, symbol):
        ## pylint: disable=method-hidden
        ok = self._ok_alpha
        if ok:
            try:
                return symbol in ok
            except TypeError:
                return False
        return True

    def _feed(self, symbol):
        stk  = self._stack
        old  = stk[-1]
        tbl  = self._xtable[old]
        while True:
            info = tbl.get(symbol, None)
            if info is not None:
                break
            info = tbl.get(AllSymbols, None)
            if info is not None:
                break
            info = self.default.__func__, _Absent
            break
        function, end_state = info
        self._next = None
        self._symbol = symbol
        ret = function(self)
        new = self._next if ret is None else ret
        new = end_state  if ret is None else ret
        if new is _Absent:
            new = None
        else:
            if not self._check_state(new):
                self._error("bad next state")
        if new is not None:
            stk[-1] = new
            self._last = old
            if new in self._stops:
                self.stop()
        return ret

    def _process_states(self, states, start_state, stop_states):
        try:
            states = [s for s in states]
        except ValueError:
            self._error("states must be a sequence")
        if len(set(states)) != len(states):
            self._error("dup states")
        if start_state is _Absent:
            if not states:
                self._error("no start state provided")
            start_state = states[0]
        elif start_state not in states:
            states = [start_state] + states
        if stop_states is _Absent:
            stop_states = [ ]
        elif isinstance(stop_states, (list, tuple)):
            for s in stop_states:
                if s not in states:
                    states.append(s)
        else:
            if stop_states not in states:
                states.append(stop_states)
            stop_states = [stop_states]
        return states, start_state, stop_states

## }}}
## {{{ test code

def test():
    ## pylint: disable=no-self-use

    class MySM(SM):
        STATE_WAIT_A = 0
        STATE_WAIT_B = 1
        STATE_DONE   = 2

        STATES      = (STATE_WAIT_A, STATE_WAIT_B)
        STOP_STATES = (STATE_DONE,)

        @transition(STATE_WAIT_A, "a", STATE_WAIT_B)
        def wait_a(self):
            pass

        @transition(STATE_WAIT_A, "c")
        def stop_a(self):
            return self.STATE_DONE

        @transition(STATE_WAIT_A, EofSymbol, STATE_DONE)
        def eof_a(self):
            print("bye")
            self.stop()

        @transition(STATE_WAIT_A, AllSymbols)
        def other_a(self):
            pass

        @transition(STATE_WAIT_B, "a")
        @transition(STATE_WAIT_B, "b", STATE_WAIT_A)
        def wait_b(self):
            pass

        @transition(STATE_WAIT_B, EofSymbol)
        def eof_b(self):
            raise error("premature eof")

        @transition(STATE_WAIT_B, AllSymbols)
        def other_b(self):
            print("b", self.symbol())
            raise error("expected 'a' or 'b'")

    sm = MySM()
    sm.start()
    sm.feed("x")
    assert sm.current() == sm.STATE_WAIT_A
    sm.feed("a")
    assert sm.current() == sm.STATE_WAIT_B
    sm.feed("a")
    assert sm.current() == sm.STATE_WAIT_B
    #sm.feed(EofSymbol) ## err: premature eof
    #sm.feed("c")       ## err: expected a or b
    sm.feed("b")
    assert sm.current() == sm.STATE_WAIT_A
    #sm.feed(EofSymbol)
    sm.feed("c")
    assert sm.current() == sm.STATE_DONE
    #sm.feed("a")   ## err: sm done

    print("pass")

if __name__ == "__main__":
    test()

## }}}

## EOF
