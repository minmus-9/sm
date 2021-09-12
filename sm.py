"""
sm.py - state machine library

tested with:
    - python 2.6.6
    - pypy   2.7.13
    - pypy   2.7.18
    - python 2.7.16
    - python 2.7.18
    - pypy   3.5.3
    - pypy   3.6.12
    - python 3.6.12
    - python 3.7.3
    - python 3.8.5
    - python 3.9.5
"""

## {{{ prologue
from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace
## pylint: disable=useless-object-inheritance,super-with-arguments

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

## }}}
## {{{ support classes and functions

_Absent    = object()
AllSymbols = object()
EofSymbol  = object()

class _Stop(Exception):
    ## pylint: disable=too-few-public-methods
    "internal: used to return from feed() when stop() is called"

class sm_error(Exception):
    ## pylint: disable=too-few-public-methods
    "base class for SM errors"

class error(sm_error, SyntaxError):
    ## pylint: disable=too-few-public-methods
    "sm state machine error"

class programming_error(sm_error, RuntimeError):
    ## pylint: disable=too-few-public-methods
    "sm programming error"

def base_with_mcs(mcs):
    "return a class with the given metaclass"
    meta_name  = mcs.__name__
    class_name = "class_%s" % meta_name
    if sys.version_info[0] < 3:
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

_DeferredStateFunctionAttr = "_deferred_sm_state_info___"

## }}}
## {{{ transition() method decorator

def transition(
        start_state, symbol, end_state=_Absent
    ):
    """
    decorator to add a new state via a method defn

    class A(SSM):
        INITIAL = "init"
        FINAL   = "fin"

        ## if state is INITIAL, switch to FINAL
        ## when we are fed an "a"
        @transition(INITIAL, "a")
        def fred(self):
            return self.FINAL

        ## if state is INITIAL, switch to FINAL
        ## when we see a "b". if barney returns
        ## a valid state, that will be the new
        ## state; i.e., end_state is a default
        @transition(INITIAL, "b", end_state=FINAL)
        def barney(self):
            pass

        ## a handler can also call next().
        ## the precedence is:
        ##   - return value of transition function
        ##     if not None
        ##   - state established with .set_state()
        ##   - end_state specified in @transition()
        ## if all of these are None, the state is
        ## not changed
        @transition(INITIAL, "c")
        def wilma(self):
            self.next(FINAL)
    """
    def transition_wrapper(function):
        "bind a transition to its implementation"
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
    "metaclass for the state machine classes"

    _class_transition_map = weakref.WeakKeyDictionary()

    def __new__(cls, name, bases, attrs):
        "collection transition info for __call__"
        kls  = type.__new__(cls, name, bases, attrs)
        xits = cls._class_transition_map.setdefault(kls, { })
        for a, v in attrs.items():
            if not callable(v):
                continue
            deferred = getattr(v, _DeferredStateFunctionAttr, None)
            if deferred is None:
                continue
            delattr(v, _DeferredStateFunctionAttr)
            xits[a] = deferred
        return kls

    def __call__(cls, *args, **kw):
        "create the sm and define its initial set of transitions"
        obj = type.__call__(cls, *args, **kw)
        if not callable(getattr(obj, "add_transition", None)):
            return obj
        xits = type(cls)._class_transition_map[cls]
        for attr, state_info in xits.items():
            func = getattr(obj, attr, None)
            # an example is better than a long speech:
            #
            # def fred(what): pass
            #
            # class C(SSM):
            #     # f is a func here
            #     @transition(0, "a")
            #     def f(self, sym, extra):
            #         pass
            #
            #     # oops, it was later redefined as a non-func!
            #     f = 42
            #
            #     # or it could be redefined like this
            #     def __init__(self):
            #         self.f = 11
            #
            #     # or this
            #     f = fred
            #
            #     # or this
            #     del f
            #
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
    "base class for state machines"

    ## over status
    _RUN_STATE_INIT = 0 ## start() not yet called
    _RUN_STATE_RUN  = 1 ## running
    _RUN_STATE_DONE = 2 ## _stop() called
    _RUN_STATE_DEAD = 3 ## fatal: _error() called, sm is dead

    def __init__(self):
        self._runstate = self._RUN_STATE_INIT

    ########################################
    ## public methods

    def add_transition(
            self,
            start_state, symbol, function=_Absent, end_state=_Absent
        ):
        """
        add a transition from start_state to end_state on the
        given symbol. function will be called if specified. if
        end_state is not specified, do not change states.
        """
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

    def start(self):
        "start the state machine running"
        rs = self._runstate
        if rs == self._RUN_STATE_RUN:
            self._error("sm already initialized")
        elif rs == self._RUN_STATE_DONE:
            self._error("sm is done")
        elif rs == self._RUN_STATE_DEAD:
            self._error("sm is dead")
        self._runstate = self._RUN_STATE_RUN

    def started(self):
        "return True if we're running"
        return self._runstate == self._RUN_STATE_RUN

    def stopped(self):
        "return True if we're stopped"
        return self._runstate == self._RUN_STATE_DONE

    def transition( ## pylint: disable=too-many-arguments
            self, start_state, symbol, function=_Absent, end_state=_Absent
        ):
        """
        add a transition from start_state to end_state on the
        given symbol. function will be called if specified. if
        end_state is not specified, do not change states. if
        function is not specified, return a wrapper function;
        i.e., this method can be used as a decorator.
        """
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
        "generate fatal error if we aren't running"
        rs = self._runstate
        if rs == self._RUN_STATE_INIT:
            self._error("sm has not started")
        elif rs == self._RUN_STATE_DONE:
            self._error("sm is done")
        elif rs == self._RUN_STATE_DEAD:
            self._error("sm is dead")

    def _error(self, msg):
        "generate a fatal error"
        self._runstate = self._RUN_STATE_DEAD
        raise programming_error(msg)

    def _stop(self):
        "enter the stopped state"
        rs = self._runstate
        if rs == self._RUN_STATE_DONE:
            return
        if rs != self._RUN_STATE_RUN:
            self._error("cannot stop sm now")
        self._runstate = self._RUN_STATE_DONE

## }}}
## {{{ sm class

class SM(_SMBase):
    "state machine class"

    ## pylint: disable=too-many-instance-attributes

    ## feed() buffering
    MAX_BUFFER_COUNT   = 32
    MAX_BUFFER_SYMBOLS = 32

    ## state push limits
    MAX_PUSH_COUNT = 32

    ## you can specify default states here
    STATES      = ( )
    START_STATE = _Absent
    STOP_STATES = ( )

    ## things that cannot be states or symbols
    _NOK_STATES = { None: True, _Absent: True }
    _NOK_ALPHA  = { None: True, _Absent: True }

    def __init__(
            self, states=( ), alphabet=( ),
            start_state=_Absent, stop_states=( ),
        ):
        """
        construct a state machine with the given states, alphabet,
        start state, and stop states. states defaults to self.STATES,
        start_state defaults to self.START_STATE or the first item in
        states if self.START_STATE isn't overridden. start_state and
        the stop_states are automatically added to states.
        """
        super(SM, self).__init__()
        states, start_state, stop_states = \
            self._process_states(
                states or self.STATES,
                self.START_STATE if start_state is _Absent else start_state,
                stop_states or self.STOP_STATES
            )

        self._ok_states = { }
        self._ok_alpha  = { }
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

        self._check_run    = self._check_run
        self._check_state  = self._check_state
        self._check_symbol = self._check_symbol

    def current(self):
        "return the current state"
        if self._runstate == self._RUN_STATE_INIT:
            self._error("sm has not started")
        return self._stack[-1]

    def default(self):  ## pylint: disable=no-self-use
        "called when no other transition can be made"
        raise error("no transition defined for current symbol")

    def depth(self):
        "return the state stack depth"
        self._check_run()
        return len(self._stack)

    def feed(self, symbol=_Absent):
        "feed a symbol or list of symbols to the state machine"
        ## pylint: disable=too-many-branches
        self._check_run()
        if isinstance(symbol, (list, tuple)):
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
            except (KeyboardInterrupt, SystemExit, sm_error):
                self._bufcnt = 0
                del q[:]
                raise
            except _Stop:
                pass
            except BaseException as exc:    ## pylint: disable=broad-except
                self._error("feed exception: " + str(exc))
        finally:
            self._feeding = False

    def feed_seq(self, seq):
        "feed a sequence of symbols to the state machine"
        ## pylint: disable=unnecessary-comprehension
        self.feed([sym for sym in seq])

    def last(self):
        "return the previous state"
        self._check_run()
        return self._last

    def next(self, symbol):
        "set the next state"
        self._check_run()
        if not self._check_symbol(symbol):
            self._error("bad symbol in next()")
        self._next = symbol

    def pop(self):
        "pop and return the top entry of the state stack"
        self._check_run()
        stk = self._stack
        if len(stk) < 2:
            self._error("state stack underflow")
        return stk.pop()

    def push(self, state, return_state=_Absent):
        """
        push a state onto the state stack. this will
        be the start state for the next feed(). if
        return_state is given, replace the top of
        stack with it before pushing the new state.
        return the old top of stack.
        """
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
        "push a symbol back into the input buffer"
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
        "return a tuple of input buffer symbols"
        self._check_run()
        return tuple(self._buffer)

    def reset(self):
        "reset the state machine"
        self._check_run()
        self._stack   = [self._state0]
        self._buffer  = collections.deque()
        self._bufcnt  = 0
        self._last    = None

    def set_state(self, state):
        "set the current state and return the old state"
        self._check_run()
        if not self._check_state(state):
            self._error("bad state")
        if state in self._stops:
            self.stop()
        stk = self._stack
        old, stk[-1] = stk[-1], state
        return old

    def stop(self):
        "stop the state machine until reset() is called"
        self._stop()
        if self._feeding:
            raise _Stop()

    def symbol(self):
        "return the current symbol"
        self._check_run()
        if not self._feeding:
            self._error("symbol() called outside of feed")
        return self._symbol

    ########################################

    def _add_transition( ## pylint: disable=too-many-arguments
            self,
            start_state, symbol, function=_Absent, end_state=_Absent
        ):
        "internal: add a transition"
        tbl = self._xtable.setdefault(start_state, { })
        if symbol in tbl:
            self._error("dup sym handler")
        tbl[symbol] = (function, end_state)

    def _check_args(self, states, alphabet):
        "internal: check and save states and alphabet for contructor"
        ok  = self._ok_states
        nok = self._NOK_STATES
        for s in states:
            if s in nok:
                self._error("illegal state")
            try:
                ok[s] = True
            except TypeError:
                self._error("unhashable state")
        try:
            ## pylint: disable=unnecessary-comprehension
            alphabet = [s for s in alphabet]
        except ValueError:
            self._error("alphabet must be a sequence")
        ok  = self._ok_alpha
        nok = self._NOK_ALPHA
        for s in alphabet:
            if s in nok:
                self._error("illegal symbol")
            try:
                ok[s] = True
            except TypeError:
                self._error("unhashable symbol")

    def _check_state(self, state):
        "internal: make sure state is good"
        ## pylint: disable=method-hidden
        if state in self._NOK_STATES:
            return False
        ok = self._ok_states
        if ok:
            try:
                return state in ok
            except TypeError:
                return False    ## not hashable
        return True

    def _check_symbol(self, symbol):
        "internal: make sure symbol is good"
        ## pylint: disable=method-hidden
        if symbol in self._NOK_ALPHA:
            return False
        ok = self._ok_alpha
        if ok:
            try:
                return symbol in ok
            except TypeError:
                return False
        return True

    def _feed(self, symbol):
        "internal: feed a single symbol to the state machine"
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
            func = getattr(self.default, "__func__", None)
            if func is None:
                raise error("no transition defined for symbol")
            info = func, _Absent
            break
        function, end_state = info
        self._next = None
        self._symbol = symbol
        ret = function(self)
        new = self._next if ret is None else ret
        new = end_state  if new is None else new
        if new is _Absent:
            new = None
        elif not self._check_state(new):
            self._error("bad next state")
        if new is not None:
            stk[-1] = new
            self._last = old
            if new in self._stops:
                self.stop()
        return ret

    def _process_states(self, states, start_state, stop_states):
        "internal: handle raw constructor arguments"
        try:
            ## pylint: disable=unnecessary-comprehension
            st = [s for s in states]
        except ValueError:
            self._error("states must be iterable")
        states = set(st)
        if len(states) != len(st):
            self._error("dup states")
        if start_state is _Absent:
            if not states:
                self._error("no start state provided")
            start_state = st[0]
        elif start_state not in states:
            states.add(start_state)
        if stop_states is _Absent:
            stop_states = [ ]
        elif isinstance(stop_states, (list, tuple)):
            for s in stop_states:
                states.add(s)
        else:
            if stop_states not in states:
                states.add(stop_states)
            stop_states = [stop_states]
        return states, start_state, stop_states

## }}}
## {{{ test code

def test():
    "test code"
    ## pylint: disable=no-self-use

    class MySM(SM):
        "demo"

        ## states
        STATE_WAIT_A = 0
        STATE_WAIT_B = 1
        STATE_DONE   = 2

        ## defaults for constructor
        START_STATE = STATE_WAIT_A
        STATES      = (STATE_WAIT_A, STATE_WAIT_B, STATE_DONE)
        STOP_STATES = (STATE_DONE,)

        @transition(STATE_WAIT_A, "a", STATE_WAIT_B)
        def wait_a(self):
            "a: wait for an 'a'"

        @transition(STATE_WAIT_A, "c")
        def stop_a(self):
            "a: stop on 'c'"
            self.next(self.STATE_DONE)
            ## or: return self.STATE_DONE

        @transition(STATE_WAIT_A, EofSymbol, STATE_DONE)
        def eof_a(self):
            "a: stop on eof"
            print("bye")
            self.stop()

        @transition(STATE_WAIT_A, AllSymbols)
        def other_a(self):
            "a: ignore anything else"

        @transition(STATE_WAIT_B, "a")
        @transition(STATE_WAIT_B, "b", STATE_WAIT_A)
        def wait_b(self):
            "b: ignore 'a', go to 'a' on 'b'"

        @transition(STATE_WAIT_B, EofSymbol)
        def eof_b(self):
            "b: error on eof"
            raise error("premature eof")

        @transition(STATE_WAIT_B, AllSymbols)
        def other_b(self):
            "b: error unless 'a' or 'b'"
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

    class SM2(SM):
        "for contructor testing"

    SM2((), "abc", start_state="u", stop_states="v")
    SM2(("u",), "abc", stop_states="v")

    print("pass")

if __name__ == "__main__":
    test()

## }}}

## EOF
