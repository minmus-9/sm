"sm.py example: http request parser"

## {{{ prologue

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

from sm import (
    AllSymbols, EofSymbol, SM, transition
)

__all__ = [
    "HttpRequest",
    "HttpResponse",
    "HttpRequestParser", "http_request_parser",
]

PY3 = sys.version_info[0] > 2

if PY3:
    ## pylint: disable=no-name-in-module,import-error
    from urllib.parse import parse_qsl, unquote, unquote_plus, urlparse
    from http.client  import responses as httpresponses
else:
    ## pylint: disable=no-name-in-module,import-error
    from urllib   import unquote, unquote_plus  ## pylint: disable=ungrouped-imports
    from urlparse import parse_qsl, urlparse
    from httplib  import responses as httpresponses

## }}}
## {{{ HttpRequestParser class

class HttpRequestParser(SM):
    "http request parser"
    ## pylint: disable=no-self-use,too-many-instance-attributes

    STATE_INIT    = "init"
    STATE_WS      = "ws"
    STATE_CR      = "cr"
    STATE_EOL     = "eol"
    STATE_METHOD  = "method"
    STATE_URI     = "uri"
    STATE_VERSION = "version"
    STATE_HEADERS = "headers"
    STATE_H_NAME  = "hdrname"
    STATE_H_VALUE = "hdrval"
    STATE_PAYLOAD = "payload"
    STATE_DONE    = "done"

    STATES = (
        STATE_INIT,
        STATE_WS,
        STATE_CR,
        STATE_EOL,
        STATE_METHOD,
        STATE_URI,
        STATE_VERSION,
        STATE_HEADERS,
        STATE_H_NAME,
        STATE_H_VALUE,
        STATE_PAYLOAD,
        STATE_DONE,
    )

    def __init__(self, lock):
        super(HttpRequestParser, self).__init__(lock)
        self.method  = ""
        self.uri     = ""
        self.version = ""
        self.headers = [ ]
        self.hnbuf   = ""
        self.hvbuf   = ""
        self.contlen = None
        self.payload = ""

    ########################################
    ## whitespace subparser
    @transition(STATE_WS, " ")
    @transition(STATE_WS, "\t")
    def _ws_ws(self):
        pass

    @transition(STATE_WS, AllSymbols)
    def _end_ws(self):
        self.putback()
        self.pop()

    ########################################
    ## end of line subparser
    @transition(STATE_EOL, "\r", end_state=STATE_CR)
    def _eol_cr(self):
        pass

    @transition(STATE_EOL, "\n")
    def _eol_nl(self):
        self.pop()

    @transition(STATE_EOL, AllSymbols)
    def _eof_rest(self):
        self.putback()
        self.pop()

    @transition(STATE_CR, "\n")
    def _eol_crlf(self):
        self.pop()

    @transition(STATE_CR, AllSymbols)
    def _cr_error(self, sym):
        raise ValueError("STATE_CR bad sym " + repr(self.symbol()))

    ALL_WS = [c for c in "\t\n\r "]
    WS     = [c for c in "\t "]
    EOL    = [c for c in "\n\r"]

    ########################################
    ## start of requst line
    @transition(STATE_INIT, symbol=ALL_WS)
    def _rqline_bad_ws(self):
        raise ValueError("STATE_INIT bad sym " + repr(self.symbol()))

    @transition(STATE_INIT, AllSymbols, end_state=STATE_METHOD)
    def _init_method(self):
        self.putback()

    ########################################
    ## request method
    @transition(STATE_METHOD, symbol=WS)
    def _method_uri(self):
        self.putback()
        self.push(self.STATE_WS, return_state=self.STATE_URI)

    @transition(STATE_METHOD, symbol=EOL)
    def _method_eol(self):
        raise ValueError("eol parsing request method")

    @transition(STATE_METHOD, AllSymbols)
    def _method_accum(self):
        self.method += self.symbol()

    ########################################
    ## uri
    @transition(STATE_URI, symbol=WS)
    def _uri_vers(self):
        self.putback()
        self.push(self.STATE_WS, return_state=self.STATE_VERSION)

    @transition(STATE_URI, symbol=EOL)
    def _uri_eol(self):
        self.version = "HTTP/0.9"
        return self.STATE_HEADERS

    @transition(STATE_URI, AllSymbols)
    def _uri_accum(self):
        self.uri += self.symbol()

    ########################################
    ## request line version
    @transition(STATE_VERSION, symbol=WS)
    def _vers_ws(self):
        raise ValueError("whitespace parsing version")

    @transition(STATE_VERSION, symbol=EOL)
    def _vers_eol(self):
        self.putback()
        self.push(self.STATE_EOL, return_state=self.STATE_HEADERS)

    @transition(STATE_VERSION, AllSymbols)
    def _vers_accum(self):
        self.version += self.symbol()

    ########################################
    ## header name subparser
    @transition(STATE_H_NAME, ":")
    def _hname_colon(self):
        if not self.hnbuf:
            raise ValueError("empty header name")
        self.push(self.STATE_WS, return_state=self.STATE_H_VALUE)

    @transition(STATE_H_NAME, symbol=ALL_WS)
    def _hname_ws(self):
        raise ValueError("whitespace parsing header name")

    @transition(STATE_H_NAME, AllSymbols)
    def _hname_accum(self):
        self.hnbuf += self.symbol()

    ########################################
    ## header value subparser
    @transition(STATE_H_VALUE, symbol=EOL)
    def _hval_eol(self):
        self.headers.append((self.hnbuf, self.hvbuf))
        self.hnbuf = self.hvbuf = ""
        self.putback()
        self.push(self.STATE_EOL, return_state=self.STATE_HEADERS)

    @transition(STATE_H_VALUE, AllSymbols)
    def _hval_accum(self):
        self.hvbuf += self.symbol()

    ########################################
    ## headers
    @transition(STATE_HEADERS, symbol=EOL)
    def _hdrs_eol(self):
        for k, v in self.headers:
            if k.lower() == "content-length":
                self.contlen = int(v)
        self.putback()
        self.push(self.STATE_EOL, return_state=self.STATE_PAYLOAD)

    @transition(STATE_HEADERS, AllSymbols)
    def _hdrs_accum(self):
        self.putback()
        return self.STATE_H_NAME

    ########################################
    ## payload
    @transition(STATE_PAYLOAD, AllSymbols)
    def _pld(self):
        if self.contlen is None:
            return self.STATE_DONE
        if len(self.payload) == self.contlen:
            return self.STATE_DONE
        self.payload += self.symbol()
        return None

    ########################################
    ## done
    @transition(STATE_DONE, AllSymbols)
    def _done(self):
        raise ValueError("trailing garbage " + repr(self.symbol()))

## }}}
## {{{ http_request_parser()

def http_request_parser(lock):
    "request parser returns (feeder(), closer())"
    p = HttpRequestParser(lock)
    p.start()
    return p.feed, p

## }}}
## {{{ test code

def test():
    "test code"
    ## pylint: disable=line-too-long
    feeder, parser = http_request_parser(None)
    feeder(symbol=[c for c in
    """
GET /index.html;x=1&y=2?a=b%20c&d#frag HTTP/1.1
Host: localhost:1234
Connection: keep-alive
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (X11; CrOS x86_64 13310.59.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.84 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Sec-Fetch-Site: none
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9

""".lstrip()])
    feeder(EofSymbol)
    d = { }
    for k in dir(parser):
        v = getattr(parser, k)
        if k.startswith("_") or callable(v) or k.lower() != k:
            continue
        d[k] = v
    from pprint import pprint
    pprint(d)

    print("pass")

## }}}

if __name__ == "__main__":
    test()

## EOF
