import copy
import math
import operator
from functools import update_wrapper

from .wsgi import ClosingIterator

# Each thread has its own greenlet, use that as the identifier for the
# context. If greenlets are not available fall back to the current
# thread ident.
try:
    from greenlet import getcurrent as get_ident
except ImportError:
    from threading import get_ident


def release_local(local):
    """Releases the contents of the local for the current context.
    This makes it possible to use locals without a manager.

    Example::

        >>> loc = Local()
        >>> loc.foo = 42
        >>> release_local(loc)
        >>> hasattr(loc, 'foo')
        False

    With this function one can release :class:`Local` objects as well
    as :class:`LocalStack` objects.  However it is not possible to
    release data held by proxies that way, one always has to retain
    a reference to the underlying local object in order to be able
    to release it.

    .. versionadded:: 0.6.1
    """
    local.__release_local__()


class Local:
    __slots__ = ("__storage__", "__ident_func__")

    def __init__(self):
        object.__setattr__(self, "__storage__", {})
        object.__setattr__(self, "__ident_func__", get_ident)

    def __iter__(self):
        return iter(self.__storage__.items())

    def __call__(self, proxy):
        """Create a proxy for a name."""
        return LocalProxy(self, proxy)

    def __release_local__(self):
        self.__storage__.pop(self.__ident_func__(), None)

    def __getattr__(self, name):
        try:
            return self.__storage__[self.__ident_func__()][name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        ident = self.__ident_func__()
        storage = self.__storage__
        try:
            storage[ident][name] = value
        except KeyError:
            storage[ident] = {name: value}

    def __delattr__(self, name):
        try:
            del self.__storage__[self.__ident_func__()][name]
        except KeyError:
            raise AttributeError(name)


class LocalStack:
    """This class works similar to a :class:`Local` but keeps a stack
    of objects instead.  This is best explained with an example::

        >>> ls = LocalStack()
        >>> ls.push(42)
        >>> ls.top
        42
        >>> ls.push(23)
        >>> ls.top
        23
        >>> ls.pop()
        23
        >>> ls.top
        42

    They can be force released by using a :class:`LocalManager` or with
    the :func:`release_local` function but the correct way is to pop the
    item from the stack after using.  When the stack is empty it will
    no longer be bound to the current context (and as such released).

    By calling the stack without arguments it returns a proxy that resolves to
    the topmost item on the stack.

    .. versionadded:: 0.6.1
    """

    def __init__(self):
        self._local = Local()

    def __release_local__(self):
        self._local.__release_local__()

    @property
    def __ident_func__(self):
        return self._local.__ident_func__

    @__ident_func__.setter
    def __ident_func__(self, value):
        object.__setattr__(self._local, "__ident_func__", value)

    def __call__(self):
        def _lookup():
            rv = self.top
            if rv is None:
                raise RuntimeError("object unbound")
            return rv

        return LocalProxy(_lookup)

    def push(self, obj):
        """Pushes a new item to the stack"""
        rv = getattr(self._local, "stack", None)
        if rv is None:
            self._local.stack = rv = []
        rv.append(obj)
        return rv

    def pop(self):
        """Removes the topmost item from the stack, will return the
        old value or `None` if the stack was already empty.
        """
        stack = getattr(self._local, "stack", None)
        if stack is None:
            return None
        elif len(stack) == 1:
            release_local(self._local)
            return stack[-1]
        else:
            return stack.pop()

    @property
    def top(self):
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
        try:
            return self._local.stack[-1]
        except (AttributeError, IndexError):
            return None


class LocalManager:
    """Local objects cannot manage themselves. For that you need a local
    manager.  You can pass a local manager multiple locals or add them later
    by appending them to `manager.locals`.  Every time the manager cleans up,
    it will clean up all the data left in the locals for this context.

    The `ident_func` parameter can be added to override the default ident
    function for the wrapped locals.

    .. versionchanged:: 0.6.1
       Instead of a manager the :func:`release_local` function can be used
       as well.

    .. versionchanged:: 0.7
       `ident_func` was added.
    """

    def __init__(self, locals=None, ident_func=None):
        if locals is None:
            self.locals = []
        elif isinstance(locals, Local):
            self.locals = [locals]
        else:
            self.locals = list(locals)
        if ident_func is not None:
            self.ident_func = ident_func
            for local in self.locals:
                object.__setattr__(local, "__ident_func__", ident_func)
        else:
            self.ident_func = get_ident

    def get_ident(self):
        """Return the context identifier the local objects use internally for
        this context.  You cannot override this method to change the behavior
        but use it to link other context local objects (such as SQLAlchemy's
        scoped sessions) to the Werkzeug locals.

        .. versionchanged:: 0.7
           You can pass a different ident function to the local manager that
           will then be propagated to all the locals passed to the
           constructor.
        """
        return self.ident_func()

    def cleanup(self):
        """Manually clean up the data in the locals for this context.  Call
        this at the end of the request or use `make_middleware()`.
        """
        for local in self.locals:
            release_local(local)

    def make_middleware(self, app):
        """Wrap a WSGI application so that cleaning up happens after
        request end.
        """

        def application(environ, start_response):
            return ClosingIterator(app(environ, start_response), self.cleanup)

        return application

    def middleware(self, func):
        """Like `make_middleware` but for decorating functions.

        Example usage::

            @manager.middleware
            def application(environ, start_response):
                ...

        The difference to `make_middleware` is that the function passed
        will have all the arguments copied from the inner application
        (name, docstring, module).
        """
        return update_wrapper(self.make_middleware(func), func)

    def __repr__(self):
        return f"<{type(self).__name__} storages: {len(self.locals)}>"


def _r_op(proxy, other, name, l_op):
    co = proxy._get_current_object()
    r_op = getattr(co, name, None)

    if r_op is not None:
        out = r_op(other)

        if out is NotImplemented:
            return l_op(other, co)

        return out

    return l_op(other, co)


class LocalProxy:
    """Acts as a proxy for a werkzeug local.  Forwards all operations to
    a proxied object.  The only operations not supported for forwarding
    are right handed operands and any kind of assignment.

    Example usage::

        from werkzeug.local import Local
        l = Local()

        # these are proxies
        request = l('request')
        user = l('user')


        from werkzeug.local import LocalStack
        _response_local = LocalStack()

        # this is a proxy
        response = _response_local()

    Whenever something is bound to l.user / l.request the proxy objects
    will forward all operations.  If no object is bound a :exc:`RuntimeError`
    will be raised.

    To create proxies to :class:`Local` or :class:`LocalStack` objects,
    call the object as shown above.  If you want to have a proxy to an
    object looked up by a function, you can (as of Werkzeug 0.6.1) pass
    a function to the :class:`LocalProxy` constructor::

        session = LocalProxy(lambda: get_current_request().session)

    .. versionchanged:: 0.6.1
       The class can be instantiated with a callable.
    """

    __slots__ = ("__local", "__name__", "__wrapped__")

    def __init__(self, local, name=None):
        object.__setattr__(self, "_LocalProxy__local", local)
        object.__setattr__(self, "__name__", name)

        if callable(local) and not hasattr(local, "__release_local__"):
            # "local" is a callable that is not an instance of Local or
            # LocalManager: mark it as a wrapped function.
            object.__setattr__(self, "__wrapped__", local)

    def _get_current_object(self):
        """Return the current object.  This is useful if you want the real
        object behind the proxy at a time for performance reasons or because
        you want to pass the object into a different context.
        """
        if not hasattr(self.__local, "__release_local__"):
            return self.__local()

        try:
            return getattr(self.__local, self.__name__)
        except AttributeError:
            raise RuntimeError(f"no object bound to {self.__name__}")

    def __repr__(self):
        try:
            obj = self._get_current_object()
        except RuntimeError:
            return f"<{type(self).__name__} unbound>"

        return repr(obj)

    def __str__(self):
        return str(self._get_current_object())

    def __bytes__(self):
        return bytes(self._get_current_object())

    def __format__(self, format_spec):
        return self._get_current_object().__format__(format_spec=format_spec)

    def __lt__(self, other):
        return self._get_current_object() < other

    def __le__(self, other):
        return self._get_current_object() <= other

    def __eq__(self, other):
        return self._get_current_object() == other

    def __ne__(self, other):
        return self._get_current_object() != other

    def __gt__(self, other):
        return self._get_current_object() > other

    def __ge__(self, other):
        return self._get_current_object() >= other

    def __hash__(self):
        return hash(self._get_current_object())

    def __bool__(self):
        try:
            return bool(self._get_current_object())
        except RuntimeError:
            return False

    def __getattr__(self, name):
        return getattr(self._get_current_object(), name)

    # __getattribute__

    def __setattr__(self, name, value):
        setattr(self._get_current_object(), name, value)

    def __delattr__(self, name):
        delattr(self._get_current_object(), name)

    def __dir__(self):
        try:
            return dir(self._get_current_object())
        except RuntimeError:
            return []

    # __get__
    # __set__
    # __delete__
    # __set_name__
    # __objclass__

    # __slots__
    # __dict__
    # __weakref__

    # __init_subclass__

    # __prepare__

    # __instancecheck__
    # __subclasscheck__

    # __class_getitem__

    def __call__(self, *args, **kwargs):
        return self._get_current_object()(*args, **kwargs)

    def __len__(self):
        return len(self._get_current_object())

    # __length_hint__

    def __getitem__(self, key):
        return self._get_current_object()[key]

    def __setitem__(self, key, value):
        self._get_current_object()[key] = value

    def __delitem__(self, key):
        del self._get_current_object()[key]

    # __missing__

    def __iter__(self):
        return iter(self._get_current_object())

    def __next__(self):
        return next(self._get_current_object())

    def __reversed__(self):
        return reversed(self._get_current_object())

    def __contains__(self, item):
        return item in self._get_current_object()

    def __add__(self, other):
        return self._get_current_object() + other

    def __sub__(self, other):
        return self._get_current_object() - other

    def __mul__(self, other):
        return self._get_current_object() * other

    def __matmul__(self, other):
        return self._get_current_object() @ other

    def __truediv__(self, other):
        return self._get_current_object() / other

    def __floordiv__(self, other):
        return self._get_current_object() // other

    def __mod__(self, other):
        return self._get_current_object() % other

    def __divmod__(self, other):
        return divmod(self._get_current_object(), other)

    def __pow__(self, other, modulo=None):
        if modulo is None:
            return self._get_current_object() ** other

        return pow(self._get_current_object(), other, modulo)

    def __lshift__(self, other):
        return self._get_current_object() << other

    def __rshift__(self, other):
        return self._get_current_object() >> other

    def __and__(self, other):
        return self._get_current_object() & other

    def __xor__(self, other):
        return self._get_current_object() ^ other

    def __or__(self, other):
        return self._get_current_object() | other

    def __radd__(self, other):
        return _r_op(self, other, "__radd__", operator.add)

    def __rsub__(self, other):
        return _r_op(self, other, "__rsub__", operator.sub)

    def __rmul__(self, other):
        return _r_op(self, other, "__rmul__", operator.mul)

    def __rmatmul__(self, other):
        return _r_op(self, other, "__rmatmul__", operator.matmul)

    def __rtruediv__(self, other):
        return _r_op(self, other, "__rtruediv__", operator.truediv)

    def __rfloordiv__(self, other):
        return _r_op(self, other, "__rfloordiv__", operator.floordiv)

    def __rmod__(self, other):
        return _r_op(self, other, "__rmod__", operator.mod)

    def __rdivmod__(self, other):
        return _r_op(self, other, "__rdivmod__", divmod)

    def __rpow__(self, other, modulo=None):
        return _r_op(self, other, "__rpow__", pow)

    def __rlshift__(self, other):
        return _r_op(self, other, "__rlshift__", operator.lshift)

    def __rrshift__(self, other):
        return _r_op(self, other, "__rrshift__", operator.rshift)

    def __rand__(self, other):
        return _r_op(self, other, "__rand__", operator.and_)

    def __rxor__(self, other):
        return _r_op(self, other, "__rxor__", operator.xor)

    def __ror__(self, other):
        return _r_op(self, other, "__ror__", operator.or_)

    def __iadd__(self, other):
        operator.iadd(self._get_current_object(), other)
        return self

    def __isub__(self, other):
        operator.isub(self._get_current_object(), other)
        return self

    def __imul__(self, other):
        operator.imul(self._get_current_object(), other)
        return self

    def __imatmul__(self, other):
        operator.imatmul(self._get_current_object(), other)
        return self

    def __itruediv__(self, other):
        operator.itruediv(self._get_current_object(), other)
        return self

    def __ifloordiv__(self, other):
        operator.ifloordiv(self._get_current_object(), other)
        return self

    def __imod__(self, other):
        operator.imod(self._get_current_object(), other)
        return self

    def __ipow__(self, other, modulo=None):
        operator.ipow(self._get_current_object(), other)
        return self

    def __ilshift__(self, other):
        operator.ilshift(self._get_current_object(), other)
        return self

    def __irshift__(self, other):
        operator.irshift(self._get_current_object(), other)
        return self

    def __iand__(self, other):
        operator.iand(self._get_current_object(), other)
        return self

    def __ixor__(self, other):
        operator.ixor(self._get_current_object(), other)
        return self

    def __ior__(self, other):
        operator.ior(self._get_current_object(), other)
        return self

    def __neg__(self):
        return -(self._get_current_object())

    def __pos__(self):
        return +(self._get_current_object())

    def __abs__(self):
        return abs(self._get_current_object())

    def __invert__(self):
        return ~(self._get_current_object())

    def __complex__(self):
        return complex(self._get_current_object())

    def __int__(self):
        return int(self._get_current_object())

    def __float__(self):
        return float(self._get_current_object())

    def __index__(self):
        return self._get_current_object().__index__()

    def __round__(self, ndigits=None):
        return round(self._get_current_object(), ndigits)

    def __trunc__(self):
        return math.trunc(self._get_current_object())

    def __floor__(self):
        return math.floor(self._get_current_object())

    def __ceil__(self):
        return math.ceil(self._get_current_object())

    def __enter__(self):
        return self._get_current_object().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._get_current_object().__exit__(exc_type, exc_value, traceback)

    # __await__
    # __aiter__
    # __anext__
    # __aenter__

    def __copy__(self):
        return copy.copy(self._get_current_object())

    def __deepcopy__(self, memo):
        return copy.deepcopy(self._get_current_object(), memo)

    # __getnewargs_ex__
    # __getnewargs__
    # __getstate__
    # __setstate__
    # __reduce__
    # __reduce_ex__
